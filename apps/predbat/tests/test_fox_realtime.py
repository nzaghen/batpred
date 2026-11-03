"""Regression tests for website live status and independent OpenAPI fallback."""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import aiohttp
from fox import FoxAPI, FoxCloudStream, FOX_CLOUD_MAX_AGE, fox_cloud_headers, fox_cloud_values, fox_cloud_planner_values
from tests.test_fox_api import MockFoxAPIWithRequests


def live_frame(soc="54.5", age=5):
    """Build the v2 website's per-plant flow payload, with mixed power units."""
    return {
        "errno": 0,
        "result": {
            "timeDiff": age,
            "lastUpdateDate": "2026-11-03 12:00:00",
            "node": {
                "bat": {"soc": soc, "charge": 1, "power": {"value": "2000", "unit": "W"}},
                "solar": {"power": {"value": "3.5", "unit": "kW"}},
                "load": {"power": {"value": "1800", "unit": "W"}},
                "grid": {"gridStatus": 2, "power": {"value": "300", "unit": "W"}},
            },
        },
    }


def make_fox():
    """Use the existing Fox double with real publishing and OpenAPI parsing."""
    fox = MockFoxAPIWithRequests()
    fox.cloud_username = "test-user"
    fox.cloud_password = "test-password"
    fox.cloud_streams = {}
    fox.device_list = [{"deviceSN": "SN1"}]
    fox.device_detail = {"SN1": {"stationID": "plant1", "hasBattery": True}}
    fox.device_values = {
        "SN1": {
            "soc": {"value": 40, "name": "SoC", "unit": "%"},
            "SoC_1": {"value": 42, "name": "SoC 1", "unit": "%"},
            "feedin": {"value": 1234.5, "name": "Export energy", "unit": "kWh"},
        }
    }
    for item in fox.device_values["SN1"].values():
        item["timestamp"] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    return fox


def context(value):
    """Build an asynchronous context manager for an HTTP session or response."""
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=value)
    manager.__aexit__ = AsyncMock(return_value=False)
    return manager


class TestFoxLiveMapping(unittest.TestCase):
    """Check the actual portal field names, units, direction and freshness."""

    def test_signed_planner_power_requires_complete_valid_live_pairs(self):
        """Never combine missing live legs with delayed OpenAPI values or invent zero."""
        base = {"batChargePower": {"value": 2000, "unit": "W"}, "batDischargePower": {"value": 0, "unit": "kW"}}
        original = deepcopy(base)
        self.assertEqual(fox_cloud_planner_values(base)["invBatPower"]["value"], -2)
        self.assertEqual(base, original)
        for bad in (None, {"value": 1, "unit": "MW"}, {"value": float("nan"), "unit": "kW"}, {"value": -1, "unit": "kW"}):
            with self.subTest(bad=bad):
                values = dict(base, batDischargePower=bad)
                self.assertNotIn("invBatPower", fox_cloud_planner_values(values))

    def test_native_live_net_power_takes_priority(self):
        """Retain direct signed net-power telemetry, including its case and units."""
        values = {"INVBATPOWER": {"value": -1234, "unit": "W"}, "batChargePower": {"value": 2, "unit": "kW"}, "batDischargePower": {"value": 0, "unit": "kW"}}
        self.assertEqual(fox_cloud_planner_values(values), values)

    def test_login_signature_matches_portal_wasm(self):
        """Pin a signature generated independently with Fox's original WASM."""
        headers = fox_cloud_headers("/basic/v0/user/login", "1793714400000", timezone.utc)
        self.assertEqual(headers["signature"], "c99eb26e5df875dfb9fa2d356bb7c7b5.5245784")
        self.assertEqual(headers["token"], "")
        self.assertEqual(headers["X-Auth-Platform"], "FOX_CLOUD_WEB_2_0")

    def test_live_soc_and_mixed_power_units(self):
        """Live SoC and power are mapped without inventing energy counters."""
        values, age = fox_cloud_values(live_frame())
        self.assertEqual(age, 5)
        for key, expected in {"SoC": 54.5, "batChargePower": 2, "batDischargePower": 0, "pvPower": 3.5, "loadsPower": 1.8, "gridConsumptionPower": 0.3, "feedinPower": 0}.items():
            self.assertEqual(values[key]["value"], expected, key)
            self.assertEqual(values[key]["source"], "fox_cloud")
        self.assertNotIn("feedin", values)

    def test_export_and_discharge_directions(self):
        """Match the website enums: battery 2=discharge, grid 1=export."""
        frame = live_frame(soc=0)
        frame["result"]["node"]["bat"]["charge"] = "2"
        frame["result"]["node"]["grid"]["gridStatus"] = "1"
        values, _ = fox_cloud_values(frame)
        self.assertEqual(values["SoC"]["value"], 0)
        self.assertEqual(values["batChargePower"]["value"], 0)
        self.assertEqual(values["batDischargePower"]["value"], 2)
        self.assertEqual(values["feedinPower"]["value"], 0.3)
        self.assertEqual(values["gridConsumptionPower"]["value"], 0)

    def test_invalid_frames_are_not_current(self):
        """Cached, missing-age, malformed and error frames cannot renew freshness."""
        for frame in [None, [], {}, {"errno": 41819}, {"errno": 0, "result": []}] + [live_frame(age=age) for age in (None, "--", -1, 30, 200, float("nan"), float("inf"), True)]:
            with self.subTest(frame=frame):
                self.assertEqual(fox_cloud_values(frame), ({}, None))

    def test_partial_values_do_not_fabricate_missing_readings(self):
        """Invalid SoC, unknown direction and hidden/unsupported power stay absent."""
        for soc in (None, "--", float("nan"), float("inf"), -1, 101, True):
            frame = live_frame(soc=soc)
            node = frame["result"]["node"]
            node["grid"]["gridStatus"] = 3
            node["bat"]["charge"] = 3
            node["solar"]["power"]["unit"] = "MW"
            node["load"]["display"] = False
            self.assertEqual(fox_cloud_values(frame)[0], {})
        frame = live_frame()
        frame["result"]["node"]["bat"]["socDisplay"] = False
        self.assertNotIn("SoC", fox_cloud_values(frame)[0])

    def test_credentials_are_optional_and_registered_as_secrets(self):
        """Website credentials leave OpenAPI authentication independent."""
        from components import COMPONENT_LIST
        from config import APPS_SCHEMA

        for name in ("cloud_username", "cloud_password"):
            option = COMPONENT_LIST["fox"]["args"][name]
            self.assertTrue(option["secret"])
            self.assertFalse(option["required"])
            self.assertIn(option["config"], APPS_SCHEMA)
        fox = make_fox()
        fox.initialize("open-key", False, cloud_username="web-user", cloud_password="web-pass")
        self.assertEqual(fox.key, "open-key")
        self.assertEqual(fox.cloud_username, "web-user")
        self.assertEqual(fox.cloud_streams, {})


class TestFoxLiveFallback(unittest.IsolatedAsyncioTestCase):
    """Exercise real publication, stale fallback and asynchronous stream lifecycle."""

    async def test_live_flow_reaches_all_automatically_bound_planner_sensors(self):
        """Auto-configured planner inputs receive live SoC and correctly signed power."""
        fox = make_fox()
        fox.device_detail["SN1"].update(hasPV=True, capacity=8, function={"scheduler": True})
        await fox.automatic_config()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        for battery_direction, grid_direction, battery_watts, grid_watts in ((1, 2, -2000, -300), (2, 1, 2000, 300)):
            with self.subTest(battery_direction=battery_direction, grid_direction=grid_direction):
                frame = live_frame(soc=0)
                frame["result"]["node"]["bat"]["charge"] = battery_direction
                frame["result"]["node"]["grid"]["gridStatus"] = grid_direction
                values, age = fox_cloud_values(frame)
                await stream.accept_values(values, age)
                for argument, expected in {"soc_percent": 0, "battery_power": battery_watts, "grid_power": grid_watts, "pv_power": 3500, "load_power": 1800}.items():
                    reading = fox.dashboard_items[fox.args_set[argument][0]]
                    value = reading["state"] * (1000 if reading["attributes"]["unit_of_measurement"] == "kW" else 1)
                    if argument == "grid_power" and fox.args_set["grid_power_invert"][0]:
                        value = -value
                    self.assertEqual(value, expected, argument)
                    self.assertEqual(reading["attributes"]["source"], "fox_cloud")
                    self.assertEqual(reading["attributes"]["website_version"], "v2")
                    self.assertTrue(reading["attributes"]["sample_age_known"])

    async def test_live_net_power_survives_openapi_poll_and_expires_to_baseline(self):
        """Fresh website net power beats an OpenAPI poll without changing its cache."""
        fox = make_fox()
        stamp = datetime.now(timezone.utc).isoformat()
        fox.device_values["SN1"].update(invBatPower={"value": 900, "unit": "W", "timestamp": stamp}, meterPower={"value": -600, "unit": "W", "timestamp": stamp})
        baseline = deepcopy(fox.device_values)
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        stream.values, _ = fox_cloud_values(live_frame())
        stream.observed_values.update(stream.values)
        stream.expires_at = float("inf")
        await fox.publish_data()
        for suffix, expected in (("invbatpower", -2), ("meterpower", 0.3)):
            self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_" + suffix]["state"], expected)
        self.assertEqual(fox.device_values, baseline)
        stream.expires_at = 0
        await fox.publish_cloud_data("SN1")
        for suffix, expected in (("invbatpower", 900), ("meterpower", -600)):
            reading = fox.dashboard_items["sensor.predbat_fox_sn1_" + suffix]
            self.assertEqual(reading["state"], expected)
            self.assertEqual(reading["attributes"]["unit_of_measurement"], "W")
            self.assertEqual(reading["attributes"]["source"], "openapi")
            self.assertNotIn("website_version", reading["attributes"])

    async def test_missing_live_net_power_does_not_reuse_an_old_frame(self):
        """A missing field retains its last known value with an explicit cached label."""
        fox = make_fox()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        stream.values, _ = fox_cloud_values(live_frame())
        stream.observed_values.update(stream.values)
        stream.expires_at = float("inf")
        await fox.publish_cloud_data("SN1")
        frame = live_frame()
        frame["result"]["node"]["bat"]["charge"] = 3
        frame["result"]["node"]["grid"]["gridStatus"] = 3
        stream.values, _ = fox_cloud_values(frame)
        await fox.publish_cloud_data("SN1")
        for suffix in ("invbatpower", "meterpower"):
            reading = fox.dashboard_items["sensor.predbat_fox_sn1_" + suffix]
            self.assertEqual(reading["state"], -2 if suffix == "invbatpower" else 0.3)
        self.assertEqual(reading["attributes"]["source"], "fox_cloud_cached")

    async def test_live_soc_wins_without_poisoning_openapi_cache(self):
        """Old per-battery SoC and later OpenAPI polls cannot overwrite live SoC."""
        fox = make_fox()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        original = deepcopy(fox.device_values)
        stream.values, _ = fox_cloud_values(live_frame(soc=0))
        stream.observed_values.update(stream.values)
        stream.expires_at = float("inf")
        await fox.publish_data()
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 0)
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["attributes"]["source"], "fox_cloud")
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_feedin"]["state"], 1234.5)
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_battery_flow"]["state"], -2)
        self.assertEqual(fox.device_values, original)
        fox.set_mock_response("/op/v1/device/real/query", [{"deviceSN": "SN1", "datas": [{"variable": "soc", "name": "SoC", "value": 35, "unit": "%"}]}])
        await FoxAPI.get_real_time_data(fox, "SN1")
        await fox.publish_data()
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 0)
        stream.expires_at = 0
        await fox.publish_cloud_data("SN1")
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 42)
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["attributes"]["source"], "openapi")
        self.assertNotIn("device_values", fox.data_age)

    async def test_live_values_expire_per_frame_and_per_device(self):
        """Partial updates fall back per field and do not affect other inverters."""
        fox = make_fox()
        fox.device_values["SN2"] = {"SoC": {"value": 87, "name": "SoC", "unit": "%", "timestamp": datetime.now(timezone.utc).isoformat()}}
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        stream.values, _ = fox_cloud_values(live_frame(soc=None))
        stream.expires_at = 125
        with patch("fox.time.monotonic", return_value=100):
            await fox.publish_data()
            self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 42)
            self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn2_soc"]["state"], 87)
            self.assertTrue(stream.current_values())
        with patch("fox.time.monotonic", return_value=125):
            self.assertEqual(stream.current_values(), {})

    async def test_login_uses_only_website_credentials(self):
        """Hash the website password and keep the OpenAPI key out of the login."""
        fox = make_fox()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        response = MagicMock()
        response.json = AsyncMock(return_value={"errno": 0, "result": {"token": "web-token"}})
        session = MagicMock()
        session.post.return_value = context(response)
        self.assertEqual(await stream.login(session), "web-token")
        kwargs = session.post.call_args.kwargs
        self.assertEqual(kwargs["json"], {"user": "test-user", "password": "dfb450efddbb5387197c84460623675b", "type": 1, "verification": 1})
        self.assertNotIn(fox.key, str(session.post.call_args))
        for data in ([], None, {"errno": 41819}, {"errno": 0, "result": {}}, {"errno": 0, "result": {"token": 123}}):
            response.json.return_value = data
            with self.assertRaises(ValueError):
                await stream.login(session)
        response.raise_for_status.side_effect = aiohttp.ClientError("HTTP failure")
        with self.assertRaises(aiohttp.ClientError):
            await stream.login(session)

    async def test_socket_ignores_stale_frames_and_updates_immediately(self):
        """Only a fresh frame publishes, and the token is safely URL encoded."""
        fox = make_fox()
        fox.publish_cloud_data = AsyncMock()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        socket = MagicMock()
        socket.send_str = AsyncMock()
        stale, fresh, closed = MagicMock(), MagicMock(), MagicMock()
        stale.type = fresh.type = aiohttp.WSMsgType.TEXT
        stale.json.return_value = live_frame(age=61)
        fresh.json.return_value = live_frame()
        closed.type = aiohttp.WSMsgType.CLOSED
        socket.receive = AsyncMock(side_effect=[stale, fresh, closed])
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(ConnectionError):
            await stream.listen(session, "secret+token&part")
        self.assertEqual(parse_qs(urlsplit(session.ws_connect.call_args.args[0]).query)["token"], ["secret+token&part"])
        self.assertIn("%2B", session.ws_connect.call_args.args[0])
        socket.send_str.assert_awaited_once_with("getdata")
        fox.publish_cloud_data.assert_awaited_once_with("SN1")
        self.assertEqual(stream.current_values()["SoC"]["value"], 54.5)

    async def test_stale_keep_alive_frames_cannot_keep_stream_alive(self):
        """Continuous old frames still hit the useful-data timeout."""
        fox = make_fox()
        fox.publish_cloud_data = AsyncMock()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        now = [100]

        async def receive():
            """Move the fake clock past the freshness deadline on a stale frame."""
            now[0] += FOX_CLOUD_MAX_AGE + 1
            message = MagicMock(type=aiohttp.WSMsgType.TEXT)
            message.json.return_value = live_frame(age=61)
            return message

        socket = MagicMock()
        socket.send_str = AsyncMock()
        socket.receive = receive
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with patch("fox.time.monotonic", side_effect=lambda: now[0]):
            with self.assertRaises(asyncio.TimeoutError):
                await stream.listen(session, "token")
        fox.publish_cloud_data.assert_not_awaited()

    async def test_failure_falls_back_and_logs_in_again_without_logging_token(self):
        """A failed reader clears live data, retries login and leaves OpenAPI intact."""
        fox = make_fox()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        stream.values, _ = fox_cloud_values(live_frame())
        stream.observed_values.update(stream.values)
        stream.expires_at = float("inf")
        stream.login = AsyncMock(return_value="secret-token")
        stream.listen = AsyncMock(side_effect=[ConnectionError("wss://host?token=secret-token"), asyncio.CancelledError()])

        with patch("fox.aiohttp.ClientSession", return_value=context(MagicMock())), patch("fox.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await stream.run()
        self.assertEqual(stream.values, {})
        self.assertEqual(stream.login.await_count, 2)
        self.assertEqual(fox.key, "test_api_key")
        self.assertNotIn("secret-token", str(fox.log_messages))
        self.assertIn("retaining freshest known status", str(fox.log_messages))
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 42)
        sleep.assert_awaited_once_with(5)

    async def test_live_publication_sends_only_flow_sensors(self):
        """Five-second updates do not republish settings, static data or counters."""
        fox = make_fox()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        stream.values, _ = fox_cloud_values(live_frame())
        stream.observed_values.update(stream.values)
        stream.expires_at = float("inf")
        fox.publish_schedule_settings_ha = AsyncMock()
        await fox.publish_cloud_data("SN1")
        self.assertEqual(len(fox.dashboard_items), 10)
        self.assertNotIn("sensor.predbat_fox_sn1_info", fox.dashboard_items)
        self.assertNotIn("sensor.predbat_fox_sn1_feedin", fox.dashboard_items)
        fox.publish_schedule_settings_ha.assert_not_awaited()
        stream.values = {}
        stream.expires_at = 0
        await fox.publish_cloud_data("SN1", refresh=True)
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_pvpower"]["state"], 3.5)
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_battery_flow"]["state"], -2)
        self.assertEqual(fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 42)

    async def test_repeated_login_failure_preserves_openapi_poll_cadence(self):
        """Bad website credentials must not make OpenAPI refresh on every retry."""
        fox = make_fox()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        fox.cloud_streams["SN1"] = stream
        fox.data_age["device_values"] = "unchanged"
        stream.login = AsyncMock(side_effect=[ValueError("login rejected"), asyncio.CancelledError()])
        with patch("fox.aiohttp.ClientSession", return_value=context(MagicMock())), patch("fox.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(asyncio.CancelledError):
                await stream.run()
        self.assertEqual(fox.data_age["device_values"], "unchanged")

    async def test_stream_start_is_optional_deduplicated_and_stopped(self):
        """Start each eligible device once and cancel all owned tasks on shutdown."""
        fox = make_fox()
        fox.cloud_password = None
        await fox.start_cloud_streams()
        self.assertEqual(fox.cloud_streams, {})
        fox.cloud_password = "password"
        with patch.object(FoxCloudStream, "run", new_callable=AsyncMock):
            await fox.start_cloud_streams()
            task = fox.cloud_streams["SN1"].task
            await fox.start_cloud_streams()
            self.assertIs(fox.cloud_streams["SN1"].task, task)
            await fox.final()
            self.assertTrue(task.done())
            self.assertEqual(fox.cloud_streams, {})
        fox.device_list.append({"deviceSN": "SN2"})
        fox.device_detail["SN2"] = {"stationID": "plant1"}
        await fox.start_cloud_streams()
        self.assertEqual(fox.cloud_streams, {})

    async def test_multiple_stream_readings_use_one_login(self):
        """The actual stream loop authenticates once for several incoming readings."""
        fox = make_fox()
        fox.publish_cloud_data = AsyncMock()
        stream = FoxCloudStream(fox, "SN1", "plant1")
        stream.login = AsyncMock(return_value="website-token")
        frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
        frame.json.return_value = live_frame()
        socket = MagicMock()
        socket.send_str = AsyncMock()
        socket.receive = AsyncMock(side_effect=[frame, frame, frame, asyncio.CancelledError()])
        session = MagicMock()
        socket_context = context(socket)
        session.ws_connect.return_value = socket_context
        session_context = context(session)
        with patch("fox.aiohttp.ClientSession", return_value=session_context):
            with self.assertRaises(asyncio.CancelledError):
                await stream.run()
        stream.login.assert_awaited_once_with(session)
        self.assertEqual(fox.publish_cloud_data.await_count, 4)  # Three readings, then cleanup
        session.ws_connect.assert_called_once()
        socket_context.__aexit__.assert_awaited_once()
        session_context.__aexit__.assert_awaited_once()


def run_fox_realtime_tests(my_predbat):
    """Register live-status regressions with Predbat's normal test runner."""
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (TestFoxLiveMapping, TestFoxLiveFallback))
    return not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
