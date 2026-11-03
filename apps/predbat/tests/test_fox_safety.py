"""Failure-injection regressions for the optional Fox website data path."""
# cspell:ignore reage battemp

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from fox import FoxCloudStream, fox_cloud_values, fox_cloud_planner_values, fox_sample_time
from tests.test_fox_realtime import make_fox, live_frame, context


class TestFoxWebsiteSafety(unittest.IsolatedAsyncioTestCase):
    """Ensure website failure cannot silently leave apparently valid live inputs."""

    def setUp(self):
        """Create actual Fox publication with independently controlled wall and monotonic clocks."""
        self.fox = make_fox()
        self.fox.device_values["SN1"].update(
            invBatPower={"value": 0.4, "unit": "kW"},
            meterPower={"value": 0.1, "unit": "kW"},
            pvPower={"value": 1, "unit": "kW"},
            loadsPower={"value": 1.5, "unit": "kW"},
        )
        self.monotonic = 100.0
        self.wall = 1_000_000.0
        for item in self.fox.device_values["SN1"].values():
            item["timestamp"] = datetime.fromtimestamp(self.wall - 60, timezone.utc).isoformat()
        self.stream = FoxCloudStream(self.fox, "SN1", "plant1")
        self.fox.cloud_streams["SN1"] = self.stream
        self.fox.base.args = {
            "soc_percent": ["sensor.predbat_fox_sn1_soc"],
            "battery_power": ["sensor.predbat_fox_sn1_invbatpower"],
            "grid_power": ["sensor.predbat_fox_sn1_meterpower"],
            "pv_power": ["sensor.predbat_fox_sn1_pvpower"],
            "load_power": ["sensor.predbat_fox_sn1_loadspower"],
        }
        self.fox.base.get_arg = self.get_arg
        self.fox.base.get_state_wrapper = lambda entity, raw=False: self.fox.base.entities.get(entity, {})
        patcher = patch("fox.time", SimpleNamespace(monotonic=lambda: self.monotonic, time=lambda: self.wall))
        patcher.start()
        self.addCleanup(patcher.stop)

    def get_arg(self, argument, default=None, index=None, **kwargs):
        """Resolve configured per-inverter entity IDs just as the real argument getter does."""
        value = self.fox.base.args.get(argument, default)
        return value[index] if isinstance(value, list) and index is not None else value

    def sync_publication(self):
        """Snapshot the real published payloads into the HA read cache."""
        self.fox.base.entities = deepcopy(self.fox.dashboard_items)

    async def publish_live(self, soc=54.5):
        """Publish a normal frame through acceptance, derivation and the actual Fox publisher."""
        values, age = fox_cloud_values(live_frame(soc=soc))
        await self.stream.accept_values(values, age)
        self.sync_publication()

    async def test_fresh_live_inputs_and_real_zero_soc_are_valid(self):
        """Zero and full SoC are real values rather than missing data or defaults."""
        for soc in (0, 54.5, 100):
            await self.publish_live(soc)
            reading = self.fox.base.entities["sensor.predbat_fox_sn1_soc"]
            self.assertEqual(reading["state"], soc)
            self.assertEqual(reading["attributes"]["valid_until"], self.wall + 25)

    async def test_fresh_inputs_allow_ordinary_execution_with_or_without_refetch(self):
        """Ordinary execution preserves its return value and execution-state persistence."""
        from predbat import PredBat

        await self.publish_live()
        base = SimpleNamespace(
            components=SimpleNamespace(get_component=lambda name: self.fox if name == "fox" else None),
            get_arg=lambda name, default=None: 1 if name == "num_inverters" else default,
            execute_plan=MagicMock(return_value=("Demand", "")),
            save_execution_state=MagicMock(),
            fetch_inverter_data=MagicMock(return_value=True),
        )
        for refetch in (False, True):
            self.assertEqual(PredBat.execute_once(base, refetch_inverter=refetch), ("Demand", ""))
        self.assertEqual(base.execute_plan.call_count, 2)
        self.assertEqual(base.save_execution_state.call_count, 2)
        base.fetch_inverter_data.assert_called_once()

    async def test_invalid_live_power_falls_back_without_poisoning_observed_values(self):
        """Malformed status must retain the latest valid net power without poisoning it."""
        self.fox.device_values["SN1"]["invBatPower"] = {"value": 0.7, "unit": "kW", "timestamp": datetime.fromtimestamp(self.wall - 60, timezone.utc).isoformat()}
        await self.publish_live()
        original = deepcopy(self.stream.observed_values["invBatPower"])
        await self.stream.accept_values({"invBatPower": {"value": 999, "unit": "MW"}, "SoC": {"value": 53, "unit": "%"}}, 0)
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_invbatpower"]
        self.assertEqual(reading["state"], -2)
        self.assertEqual(reading["attributes"]["source"], "fox_cloud_cached")
        self.assertEqual(self.stream.observed_values["invBatPower"], original)

    async def test_missing_unit_or_invalid_soc_does_not_guess_a_live_value(self):
        """Units are required and malformed values are excluded before derivation."""
        frame = live_frame()
        frame["result"]["node"]["solar"]["power"].pop("unit")
        self.assertNotIn("pvPower", fox_cloud_values(frame)[0])
        for value in (-1, 101, float("nan"), True):
            self.assertNotIn("SoC_1", fox_cloud_planner_values({"SoC_1": {"value": value, "unit": "%"}}))

    async def test_late_v2_frame_does_not_restart_a_dead_connection(self):
        """A frame delivered after the receive deadline cannot reset its freshness."""

        async def receive():
            """Simulate a delayed receive callback, such as after a blocked HA publication."""
            self.monotonic += 31
            message = MagicMock(type=aiohttp.WSMsgType.TEXT)
            message.json.return_value = live_frame()
            return message

        socket = SimpleNamespace(send_str=AsyncMock(), receive=receive)
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(asyncio.TimeoutError):
            await self.stream.listen(session, "private-token")
        self.assertEqual(self.stream.values, {})
        self.assertFalse(self.fox.dashboard_items)

    async def test_serial_filter_cannot_hide_another_inverter_in_same_plant(self):
        """A selected serial does not turn plant totals into per-inverter telemetry."""
        self.fox.inverter_sn_filter = ["SN1"]
        self.fox.set_mock_response("/op/v0/device/list", {"data": [{"deviceSN": "SN1", "stationID": "plant1"}, {"deviceSN": "SN2", "stationID": "plant1"}]})
        self.assertEqual(len(await self.fox.get_device_list()), 1)
        self.fox.cloud_streams = {}
        with patch.object(FoxCloudStream, "run", new_callable=AsyncMock) as reader:
            await self.fox.start_cloud_streams()
            reader.assert_not_called()
        self.assertEqual(self.fox.cloud_streams, {})

    async def test_complete_inventory_starts_only_selected_single_inverter_plant(self):
        """Filtering a distinct plant is safe once the complete membership is known."""
        self.fox.inverter_sn_filter = ["SN1"]
        self.fox.set_mock_response("/op/v0/device/list", {"total": 2, "data": [{"deviceSN": "SN1", "stationID": "plant1"}, {"deviceSN": "SN2", "stationID": "plant2"}]})
        await self.fox.get_device_list()
        self.fox.cloud_streams = {}
        with patch.object(FoxCloudStream, "run", new_callable=AsyncMock):
            await self.fox.start_cloud_streams()
            self.assertEqual(set(self.fox.cloud_streams), {"SN1"})
            await self.fox.final()

    async def test_incomplete_or_unknown_inventory_cannot_enable_plant_streams(self):
        """A cached subset or a truncated first page is insufficient evidence of membership."""
        for inventory in (None, [{"deviceSN": "UNKNOWN"}]):
            self.fox.cloud_device_list = inventory
            self.fox.cloud_streams = {}
            await self.fox.start_cloud_streams()
            self.assertEqual(self.fox.cloud_streams, {})
        self.fox.set_mock_response("/op/v0/device/list", {"total": 101, "data": [{"deviceSN": "SN1", "stationID": "plant1"}]})
        await self.fox.get_device_list()
        self.assertIsNone(self.fox.cloud_device_list)
        await self.fox.start_cloud_streams()
        self.assertEqual(self.fox.cloud_streams, {})

    async def test_healthy_v2_push_does_not_restart_subscription_every_twenty_seconds(self):
        """Portal-style idle keepalive leaves regular push frames uninterrupted."""
        self.fox.publish_cloud_data = AsyncMock()
        frames = iter(range(105, 166, 5))

        async def receive():
            """Provide a minute of regular server pushes, followed by socket closure."""
            tick = next(frames, None)
            if tick is None:
                return MagicMock(type=aiohttp.WSMsgType.CLOSED)
            self.monotonic = tick
            frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
            frame.json.return_value = live_frame()
            return frame

        socket = MagicMock()
        socket.send_str = AsyncMock()
        socket.receive = receive
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(ConnectionError):
            await self.stream.listen(session, "private-token")
        socket.send_str.assert_awaited_once_with("getdata")
        self.assertEqual(self.fox.publish_cloud_data.await_count, 13)

    async def test_v2_expiry_logs_numeric_frame_diagnostics_without_server_text(self):
        """Distinguish rejected frames from silence without exposing server payloads."""
        frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
        frame.json.return_value = {"errno": 41819, "result": {"timeDiff": 300}, "msg": "private-token"}
        replies = iter([frame])

        async def receive():
            """Expire after one rejected frame without waiting in real time."""
            reply = next(replies, None)
            if reply is not None:
                return reply
            self.monotonic += 31
            raise asyncio.TimeoutError()

        socket = MagicMock()
        socket.send_str = AsyncMock()
        socket.receive = receive
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(asyncio.TimeoutError):
            await self.stream.listen(session, "private-token")
        logs = str(self.fox.log_messages)
        self.assertIn("errno=41819.0, timeDiff=300.0s", logs)
        self.assertNotIn("private-token", logs)

    def date_openapi(self, sampled_at):
        """Give every baseline status field one explicit inverter sample time."""
        stamp = datetime.fromtimestamp(sampled_at, timezone.utc).isoformat()
        for item in self.fox.device_values["SN1"].values():
            item["timestamp"] = stamp

    async def test_sample_timestamp_formats_and_dst_are_unambiguous(self):
        """Parse numeric offsets and reject missing times and ambiguous local DST times."""
        expected = datetime(2026, 9, 14, 17, 43, 9, tzinfo=timezone.utc).timestamp()
        for stamp in ("2026-09-14 18:43:09 BST+0100", "2026-09-14T17:43:09Z", "2026-09-14 17:43:09 GMT+0000", "2026-09-14T18:43:09+01:00", "2026-09-14 18:43:09"):
            self.assertEqual(fox_sample_time(stamp, self.fox.local_tz), expected, stamp)
        for stamp in (None, True, 1234, "", "2026-09-14", "2026-99-99 25:00:00", "2026-10-25 01:30:00", "2026-03-29 01:30:00"):
            self.assertIsNone(fox_sample_time(stamp, self.fox.local_tz), stamp)

    async def test_openapi_only_setups_keep_their_existing_age_policy(self):
        """Optional website safety must not alter installations without website credentials."""
        self.fox.cloud_password = None
        self.date_openapi(self.wall - 86400)
        await self.fox.publish_data()
        self.sync_publication()
        reading = self.fox.base.entities["sensor.predbat_fox_sn1_soc"]
        self.assertEqual(reading["state"], 42)
        self.assertNotIn("valid_until", reading["attributes"])

    async def test_v2_silent_five_minute_feed_is_requested_before_expiry(self):
        """A quiet stream can resume before recovery expiry while data expires independently."""
        events = iter([(tick, "frame") for tick in range(100, 401, 5)] + [(425, "timeout"), (430, "timeout"), (431, "frame"), (432, "closed")])
        publications = []

        async def receive():
            """Model five minutes of pushes, then wait for a fresh getdata request."""
            tick, kind = next(events)
            self.wall += tick - self.monotonic
            self.monotonic = tick
            if kind == "timeout":
                raise asyncio.TimeoutError()
            if kind == "closed":
                return MagicMock(type=aiohttp.WSMsgType.CLOSED)
            if tick == 431:
                self.assertEqual(socket.send_str.await_count, 2)
                self.assertFalse(self.stream.current_values())
            frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
            frame.json.return_value = live_frame(soc=23)
            return frame

        async def publish(sn, refresh=False):
            """Capture real publication metadata at each accepted sample."""
            await self.fox.publish_data(live_only=True)
            publications.append(deepcopy(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]))

        self.fox.publish_cloud_data = publish
        socket = SimpleNamespace(send_str=AsyncMock(), receive=receive)
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(ConnectionError):
            await self.stream.listen(session, "private-token")
        self.assertEqual(len(publications), 63)
        self.assertEqual(socket.send_str.await_count, 2)
        self.assertEqual(publications[61]["state"], 23)
        self.assertTrue(publications[61]["attributes"]["cached"])
        self.assertTrue(all(item["state"] == 23 and item["attributes"]["source"] == "fox_cloud" for item in publications[:61] + publications[62:]))

    async def test_v2_getdata_without_response_cannot_renew_freshness(self):
        """Sending requests never makes an old numeric SoC usable for planning."""
        self.date_openapi(self.wall - 60)
        events = iter([(100, "frame"), (125, "timeout"), (130, "timeout"), (160, "timeout"), (190, "timeout"), (220, "timeout")])

        async def receive():
            """Ignore the two renewal requests and stop at the original sample deadline."""
            tick, kind = next(events)
            self.wall += tick - self.monotonic
            self.monotonic = tick
            if kind == "timeout":
                raise asyncio.TimeoutError()
            frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
            frame.json.return_value = live_frame()
            return frame

        socket = SimpleNamespace(send_str=AsyncMock(), receive=receive)
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(asyncio.TimeoutError):
            await self.stream.listen(session, "private-token")
        self.sync_publication()
        self.assertEqual(socket.send_str.await_count, 4)
        self.assertEqual(self.stream.current_values(), {})
        self.assertEqual(self.stream.expires_at, 0)
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 54.5)
        self.assertIn("getdata requests=4", str(self.fox.log_messages))

    async def test_browser_capture_stale_frames_recover_on_same_socket(self):
        """Replay the observed 88-second browser gap without accepting its cached SoC frames."""
        self.date_openapi(self.wall - 60)
        events = iter([(100, 5), (125, None), (130, None), (131.57, 61), (161.57, None), (164.27, 64), (188.29, 5), (190, "closed")])
        states = []

        async def receive():
            """Use the capture's receive gaps and portal idle-request timing."""
            tick, age = next(events)
            self.wall += tick - self.monotonic
            self.monotonic = tick
            if age is None:
                raise asyncio.TimeoutError()
            if age == "closed":
                return MagicMock(type=aiohttp.WSMsgType.CLOSED)
            frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
            frame.json.return_value = live_frame(soc=99 if age > 30 else 23, age=age)
            return frame

        async def publish(sn, refresh=False):
            """Record accepted website states and the explicit unavailable interval."""
            await self.fox.publish_data(live_only=True)
            states.append(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"])

        self.fox.publish_cloud_data = publish
        socket = SimpleNamespace(send_str=AsyncMock(), receive=receive)
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(ConnectionError):
            await self.stream.listen(session, "private-token")
        self.assertEqual(states, [23, 23, 23])
        self.assertEqual(socket.send_str.await_count, 3)
        session.ws_connect.assert_called_once()
        self.assertNotIn("heartbeat", session.ws_connect.call_args.kwargs)

    async def test_v2_preparation_is_signed_and_does_not_renew_status(self):
        """Mirror the portal's GET plantID request every sixty seconds using the v2 token."""
        await self.publish_live()
        expiry = self.stream.expires_at
        response = MagicMock()

        async def reply():
            """Spend two seconds on HTTP without changing status freshness."""
            self.monotonic += 2
            return {"errno": 0, "result": {"plantLive": 0}}

        response.json = reply
        session = MagicMock()
        session.get.return_value = context(response)
        with patch("fox.asyncio.sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError()) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await self.stream.renew_v2_flow(session, "private-token")
        call = session.get.call_args
        self.assertTrue(call.args[0].endswith("/dew/w/plant/flow/preInfo"))
        self.assertEqual(call.kwargs["params"], {"plantID": "plant1"})
        self.assertEqual(call.kwargs["headers"]["token"], "private-token")
        self.assertEqual(call.kwargs["headers"]["X-Auth-Platform"], "FOX_CLOUD_WEB_2_0")
        sleep.assert_awaited_once_with(58)
        self.assertEqual(self.stream.expires_at, expiry)

    async def test_v2_preparation_failure_is_nonfatal_and_redacted(self):
        """A rejected periodic read logs a code and retries without promoting status."""
        response = MagicMock()
        response.json = AsyncMock(return_value={"errno": 41819, "msg": "private-token"})
        session = MagicMock()
        session.get.return_value = context(response)
        with patch("fox.asyncio.sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError()) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await self.stream.renew_v2_flow(session, "private-token")
        sleep.assert_awaited_once_with(60)
        self.assertFalse(self.stream.values)
        self.assertIn("errno=41819", str(self.fox.log_messages))
        self.assertNotIn("private-token", str(self.fox.log_messages))

    async def test_latest_v2_survives_pause_until_openapi_catches_up(self):
        """A delayed API cannot replace a newer v2 SoC; a newer API sample can."""
        self.date_openapi(self.wall - 120)
        baseline = deepcopy(self.fox.device_values)
        await self.publish_live(soc=23)
        self.wall += 90
        self.monotonic += 90
        self.stream.values = {}
        self.stream.expires_at = 0
        await self.fox.publish_cloud_data("SN1", refresh=True)
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]
        self.assertEqual(reading["state"], 23)
        self.assertEqual(reading["attributes"]["source"], "fox_cloud_cached")
        self.assertTrue(reading["attributes"]["cached"])
        self.assertEqual(reading["attributes"]["sample_age_seconds"], 95)
        self.assertEqual(self.fox.device_values, baseline)
        self.date_openapi(self.wall - 10)
        self.fox.device_values["SN1"]["SoC_1"]["value"] = 22
        await self.fox.publish_cloud_data("SN1")
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]
        self.assertEqual(reading["state"], 22)
        self.assertEqual(reading["attributes"]["source"], "openapi")
        self.assertEqual(reading["attributes"]["sample_age_seconds"], 10)
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_invbatpower"]["state"], 0.4)
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_battery_flow"]["state"], 0.4)
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_batchargepower"]["state"], 0)
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_batdischargepower"]["state"], 0.4)

    async def test_newer_openapi_can_win_while_v2_is_still_live(self):
        """Choose by sample time rather than always preferring a connected website."""
        self.date_openapi(self.wall - 1)
        await self.publish_live(soc=23)
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]
        self.assertEqual(reading["state"], 42)
        self.assertEqual(reading["attributes"]["source"], "openapi")

    async def test_undated_or_invalid_openapi_does_not_replace_known_v2(self):
        """Keep the last valid dated sample when the API cannot establish a newer reading."""
        await self.publish_live(soc=23)
        self.wall += 40
        self.monotonic += 40
        for stamp in (None, "bad", datetime.fromtimestamp(self.wall + 3600, timezone.utc).isoformat()):
            for item in self.fox.device_values["SN1"].values():
                item["timestamp"] = stamp
            await self.fox.publish_cloud_data("SN1")
            self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 23)

    async def test_live_only_status_retains_actual_last_value_instead_of_zero(self):
        """A missing OpenAPI net-power field must not erase the last known website reading."""
        self.fox.device_values["SN1"].pop("meterPower")
        await self.publish_live()
        self.wall += 40
        self.monotonic += 40
        self.stream.values = {}
        await self.fox.publish_cloud_data("SN1")
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_meterpower"]
        self.assertEqual(reading["state"], 0.3)
        self.assertEqual(reading["attributes"]["source"], "fox_cloud_cached")

    async def test_energy_counters_remain_on_openapi(self):
        """Website counters must not create jumps or resets in cumulative energy history."""
        await self.stream.accept_values({"feedin": {"value": 9999, "unit": "kWh"}, "SoC": {"value": 23, "unit": "%"}}, 0)
        await self.fox.publish_data()
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_feedin"]["state"], 1234.5)
        self.assertNotIn("feedin", self.stream.observed_values)

    async def test_cached_readings_allow_ordinary_execution(self):
        """A live-data pause no longer creates a Fox-specific execution suspension."""
        from predbat import PredBat

        await self.publish_live(soc=23)
        self.wall += 90
        self.monotonic += 90
        await self.fox.publish_cloud_data("SN1")
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"], 23)
        base = SimpleNamespace(
            components=SimpleNamespace(get_component=lambda name: self.fox),
            get_arg=lambda name, default=None: 1 if name == "num_inverters" else default,
            execute_plan=MagicMock(return_value=("Demand", "")),
            save_execution_state=MagicMock(),
            fetch_inverter_data=MagicMock(return_value=True),
        )
        for refetch in (False, True):
            self.assertEqual(PredBat.execute_once(base, refetch_inverter=refetch), ("Demand", ""))
        self.assertEqual(base.execute_plan.call_count, 2)

    async def test_login_failure_uses_no_legacy_login_and_keeps_openapi(self):
        """V2 auth failures retry v2 with backoff and never try the legacy website API."""
        response = MagicMock()
        response.json = AsyncMock(side_effect=[{"errno": 41819, "msg": "private-token"}, asyncio.CancelledError()])
        session = MagicMock()
        session.post.return_value = context(response)
        with patch("fox.aiohttp.ClientSession", return_value=context(session)), patch("fox.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await self.stream.run()
        self.assertEqual(session.post.call_count, 2)
        self.assertTrue(all(call.args[0].endswith("/basic/v0/user/login") for call in session.post.call_args_list))
        sleep.assert_awaited_once_with(5)
        self.assertEqual(self.fox.key, "test_api_key")
        self.assertNotIn("private-token", str(self.fox.log_messages))

    async def test_single_retry_loop_reuses_token_and_cleans_up_periodic_tasks(self):
        """A dropped working connection retries without login and cancels each periodic task."""
        started = [asyncio.Event(), asyncio.Event()]
        cancelled = []
        attempts = []
        self.stream.login = AsyncMock(return_value="private-token")

        async def renew(session, token):
            """Track task startup and cancellation for each connection."""
            index = len(attempts) - 1
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(index)

        async def listen(session, token):
            """Drop once after valid status, then cancel the recovered connection."""
            attempts.append(token)
            await started[len(attempts) - 1].wait()
            if len(attempts) == 1:
                await self.publish_live(soc=23)
                raise ConnectionError("private-token")
            raise asyncio.CancelledError()

        self.stream.renew_v2_flow = renew
        self.stream.listen = listen
        with patch("fox.aiohttp.ClientSession", return_value=context(MagicMock())), patch("fox.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await self.stream.run()
        self.stream.login.assert_awaited_once()
        self.assertEqual(attempts, ["private-token", "private-token"])
        self.assertEqual(cancelled, [0, 1])
        sleep.assert_awaited_once_with(5)
        self.assertNotIn("private-token", str(self.fox.log_messages))


def run_fox_safety_tests(my_predbat):
    """Run website failure-injection tests through Predbat's normal test harness."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFoxWebsiteSafety)
    return not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
