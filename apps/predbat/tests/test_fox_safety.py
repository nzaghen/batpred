"""Failure-injection regressions for the optional Fox website data path."""
# cspell:ignore reage

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from execute import Execute
from fox import FoxCloudStream, fox_cloud_values, fox_cloud_planner_values, fox_sample_time, FOX_OPENAPI_MAX_AGE
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
            item["timestamp"] = datetime.fromtimestamp(self.wall, timezone.utc).isoformat()
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
        await self.stream.accept_values(values, age, "v2")
        self.sync_publication()

    async def test_fresh_live_inputs_and_real_zero_soc_are_valid(self):
        """Zero and full SoC are real values rather than missing data or defaults."""
        for soc in (0, 54.5, 100):
            await self.publish_live(soc)
            self.assertIsNone(self.fox.planner_data_error(0))
            reading = self.fox.base.entities["sensor.predbat_fox_sn1_soc"]
            self.assertEqual(reading["state"], soc)
            self.assertEqual(reading["attributes"]["valid_until"], self.wall + 25)

    async def test_stalled_publication_cannot_keep_live_readings_usable(self):
        """Use-time expiry protects planning even when the socket/publisher cannot run."""
        await self.publish_live()
        self.wall += 25
        self.monotonic += 25
        self.assertEqual(self.stream.current_values(), {})
        self.assertIn("soc_percent", self.fox.planner_data_error(0))
        self.assertIn("suspended", self.fox.planner_data_error(0))

    async def test_stale_live_data_blocks_both_full_and_quick_inverter_reads(self):
        """Both shared entry points stop before any inverter can be created or written."""
        await self.publish_live()
        self.wall += 30
        for quick in (False, True):
            base = SimpleNamespace(
                components=SimpleNamespace(get_component=lambda name: self.fox if name == "fox" else None),
                get_arg=lambda name, default=None: 1 if name == "num_inverters" else default,
                log=MagicMock(),
                record_status=MagicMock(),
                plan_valid=True,
            )
            with patch("execute.Inverter") as constructor:
                self.assertFalse(Execute.fetch_inverter_data(base, quick=quick))
                constructor.assert_not_called()
            self.assertFalse(base.plan_valid)
            self.assertIn("suspended", base.record_status.call_args.args[0])

    async def test_execution_does_not_write_after_rejected_live_inputs(self):
        """The execution wrapper returns before execute_plan on a failed fresh read."""
        from predbat import PredBat

        await self.publish_live()
        self.wall += 30
        base = SimpleNamespace(
            components=SimpleNamespace(get_component=lambda name: self.fox if name == "fox" else None),
            get_arg=lambda name, default=None: 1 if name == "num_inverters" else default,
            log=MagicMock(),
            record_status=MagicMock(),
            plan_valid=True,
            execute_plan=MagicMock(),
        )
        base.fetch_inverter_data = MethodType(Execute.fetch_inverter_data, base)
        for refetch in (False, True):
            self.assertIsNone(PredBat.execute_once(base, refetch_inverter=refetch))
            base.execute_plan.assert_not_called()

    async def test_valid_recovery_releases_planning_guard(self):
        """Fresh frames recover automatically; expired frames never become zero SoC."""
        await self.publish_live()
        self.wall += 30
        self.monotonic += 30
        self.assertIsNotNone(self.fox.planner_data_error(0))
        await self.publish_live(soc=53)
        self.assertIsNone(self.fox.planner_data_error(0))
        self.assertEqual(self.fox.base.entities["sensor.predbat_fox_sn1_soc"]["state"], 53)

    async def test_fresh_inputs_allow_ordinary_execution_with_or_without_refetch(self):
        """The new guard allows healthy execution and does not change its return or persistence."""
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

    async def test_openapi_fallback_preserves_pre_existing_behaviour(self):
        """A correctly dated fallback newer than the live sample remains usable."""
        await self.publish_live()
        baseline = deepcopy(self.fox.device_values)
        self.wall += 30
        self.monotonic += 30
        self.stream.values = {}
        await self.fox.publish_cloud_data("SN1", refresh=True)
        self.sync_publication()
        self.assertIsNone(self.fox.planner_data_error(0))
        reading = self.fox.base.entities["sensor.predbat_fox_sn1_soc"]
        self.assertEqual(reading["state"], 42)
        self.assertEqual(reading["attributes"]["source"], "openapi")
        self.assertEqual(reading["attributes"]["valid_until"], 1_000_000 + FOX_OPENAPI_MAX_AGE)
        self.assertEqual(self.fox.device_values, baseline)

    async def test_guard_does_not_change_disabled_or_external_sensor_setups(self):
        """A disabled website or a user's separately configured sensors retain old behaviour."""
        await self.publish_live()
        self.wall += 30
        self.fox.cloud_password = None
        self.assertIsNone(self.fox.planner_data_error(0))
        self.fox.cloud_password = "password"
        self.fox.base.args = {key: ["sensor.external_" + key] for key in self.fox.base.args}
        self.assertIsNone(self.fox.planner_data_error(0))

    async def test_missing_live_only_fallback_blocks_instead_of_becoming_zero(self):
        """A live-only input with no old-API counterpart is explicitly unavailable."""
        self.fox.device_values["SN1"].pop("meterPower")
        await self.publish_live()
        self.stream.values = {}
        self.stream.expires_at = 0
        await self.fox.publish_cloud_data("SN1")
        self.sync_publication()
        self.assertIsNone(self.fox.base.entities["sensor.predbat_fox_sn1_meterpower"]["state"])
        self.assertIn("grid_power", self.fox.planner_data_error(0))

    async def test_energy_counters_never_switch_sources_on_v1_fallback(self):
        """A newer v1 counter cannot jump backwards when the old API resumes."""
        await self.stream.accept_values({"feedin": {"value": 1240, "unit": "kWh"}, "SoC": {"value": 50, "unit": "%"}}, 0, "v1")
        await self.fox.publish_data()
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_feedin"]
        self.assertEqual(reading["state"], 1234.5)
        self.assertEqual(reading["attributes"]["source"], "openapi")
        self.assertNotIn("feedin", self.stream.observed_values)

    async def test_invalid_website_attributes_are_rejected_per_planner_field(self):
        """Bad state, unit or expiry cannot turn into a plausible default at read time."""
        await self.publish_live()
        original = deepcopy(self.fox.base.entities)
        for argument, entities in self.fox.base.args.items():
            for field, bad in (("state", True), ("state", float("nan")), ("state", float("inf")), ("state", 1e308), ("state", None), ("unit_of_measurement", "MW"), ("valid_until", None), ("valid_until", float("nan")), ("valid_until", self.wall)):
                with self.subTest(argument=argument, field=field, bad=bad):
                    self.fox.base.entities = deepcopy(original)
                    reading = self.fox.base.entities[entities[0]]
                    if field == "state":
                        reading["state"] = bad
                    else:
                        reading["attributes"][field] = bad
                    self.assertIn(argument, self.fox.planner_data_error(0))

    async def test_invalid_live_power_falls_back_without_poisoning_observed_values(self):
        """Malformed v1 detail cannot replace a good net power or keep it live."""
        self.fox.device_values["SN1"]["invBatPower"] = {"value": 0.7, "unit": "kW", "timestamp": datetime.fromtimestamp(self.wall, timezone.utc).isoformat()}
        await self.publish_live()
        original = deepcopy(self.stream.observed_values["invBatPower"])
        await self.stream.accept_values({"invBatPower": {"value": 999, "unit": "MW"}, "SoC": {"value": 53, "unit": "%"}}, 0, "v1")
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_invbatpower"]
        self.assertEqual(reading["state"], 0.7)
        self.assertEqual(reading["attributes"]["source"], "openapi")
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

    async def test_v1_battery_request_does_not_reage_older_power(self):
        """Waiting for a supplement preserves the receive time of existing flow readings."""

        async def request(session, token, path, data, post=False):
            """Delay only the battery response while leaving the flow's original age intact."""
            if path.endswith("/list"):
                return {"devices": [{"deviceSN": "SN1", "id": "device1"}]}
            self.monotonic += 20
            return {"batterys": [{"soc": 50}]}

        self.stream.v1_request = request
        message = MagicMock(type=aiohttp.WSMsgType.TEXT)
        message.json.return_value = {"msgType": "data", "sequence": "flow", "data": {"loadsPower": {"value": 1.2, "unit": "kW"}}}
        socket = SimpleNamespace(send_str=AsyncMock(), send_json=AsyncMock(), receive=AsyncMock(side_effect=[message, MagicMock(type=aiohttp.WSMsgType.CLOSED)]))
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(ConnectionError):
            await self.stream.listen_v1(session, "private-token")
        self.assertEqual(self.stream.expires_at, 130)
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_loadspower"]["attributes"]["valid_until"], self.wall + 10)

    async def test_v1_supplement_has_only_remaining_freshness_budget(self):
        """An HTTP read is cancelled before it can prolong stale socket data."""
        original_wait = asyncio.wait_for
        self.stream.v1_request = AsyncMock(side_effect=[{"devices": [{"deviceSN": "SN1", "id": "device1"}]}, {"batterys": [{"soc": 50}]}])

        async def receive():
            """Consume twenty seconds of the initial receive budget before a valid frame."""
            self.monotonic = 120
            message = MagicMock(type=aiohttp.WSMsgType.TEXT)
            message.json.return_value = {"msgType": "data", "sequence": "flow", "data": {"loadsPower": {"value": 1.2, "unit": "kW"}}}
            return message

        socket = SimpleNamespace(send_str=AsyncMock(), send_json=AsyncMock(), receive=receive)
        session = MagicMock()
        session.ws_connect.return_value = context(socket)

        async def bounded_wait(awaitable, timeout):
            """Verify the supplementary request's timeout and end the simulated stream."""
            if self.monotonic == 120:
                self.assertEqual(timeout, 10)
                awaitable.close()
                raise asyncio.TimeoutError()
            return await original_wait(awaitable, timeout)

        with patch("fox.asyncio.wait_for", side_effect=bounded_wait):
            with self.assertRaises(asyncio.TimeoutError):
                await self.stream.listen_v1(session, "private-token")
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

    async def test_v1_portal_device_ids_reach_socket_and_battery_request(self):
        """Use the actual list response fields, not the detail route's renamed fields."""
        self.stream.v1_request = AsyncMock(
            side_effect=[
                {"devices": [{"deviceSN": "OTHER", "deviceID": "wrong-device"}, {"deviceSN": "SN1", "deviceID": "device1", "emsID": "ems1", "wifiMeterID": "meter1"}]},
                {"batterys": [{"soc": 23}]},
            ]
        )
        frame = MagicMock(type=aiohttp.WSMsgType.TEXT)
        frame.json.return_value = {"msgType": "data", "sequence": "flow", "data": {"loadsPower": {"value": 1.2}}}
        socket = MagicMock()
        socket.send_json = AsyncMock()
        socket.send_str = AsyncMock()
        socket.receive = AsyncMock(side_effect=[frame, MagicMock(type=aiohttp.WSMsgType.CLOSED)])
        session = MagicMock()
        session.ws_connect.return_value = context(socket)
        with self.assertRaises(ConnectionError):
            await self.stream.listen_v1(session, "private-token")
        for call in socket.send_json.await_args_list:
            self.assertEqual(call.args[0]["parameters"], {"deviceID": "device1", "wifiMeterID": "meter1", "chargersSN": [], "emsID": "ems1"})
        self.stream.v1_request.assert_awaited_with(session, "private-token", "/generic/v0/device/battery/info", {"id": "device1"})
        self.assertEqual(self.stream.current_values()["SoC"]["value"], 23)

    async def test_v1_duplicate_portal_serials_cannot_bind_an_arbitrary_device(self):
        """Correct identifier spelling must not weaken exact serial matching."""
        self.stream.v1_request = AsyncMock(return_value={"devices": [{"deviceSN": "SN1", "deviceID": "one"}, {"deviceSN": "SN1", "deviceID": "two"}]})
        session = MagicMock()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            await self.stream.listen_v1(session, "private-token")
        session.ws_connect.assert_not_called()

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

    async def test_older_fallback_cannot_raise_soc_or_contaminate_history(self):
        """Reproduce a discharge-time source switch to a delayed, higher OpenAPI SoC."""
        self.date_openapi(self.wall - 120)
        original = deepcopy(self.fox.device_values)
        await self.publish_live(soc=23)
        self.wall += 30
        self.monotonic += 30
        self.stream.values = {}
        await self.fox.publish_cloud_data("SN1", refresh=True)
        self.sync_publication()
        reading = self.fox.base.entities["sensor.predbat_fox_sn1_soc"]
        self.assertIsNone(reading["state"])
        self.assertIn("predates", reading["attributes"]["stale_reason"])
        self.assertIn("suspended", self.fox.planner_data_error(0))
        self.assertIsNone(self.fox.dashboard_items["sensor.predbat_fox_sn1_battery_flow"]["state"])
        self.assertEqual(self.fox.device_values, original)
        await self.fox.publish_data()
        self.assertEqual(self.fox.dashboard_items["sensor.predbat_fox_sn1_feedin"]["state"], 1234.5)
        self.date_openapi(self.wall - 1)
        self.fox.device_values["SN1"]["SoC_1"]["value"] = 22
        await self.fox.publish_cloud_data("SN1")
        self.sync_publication()
        self.assertEqual(self.fox.base.entities["sensor.predbat_fox_sn1_soc"]["state"], 22)
        self.assertIsNone(self.fox.planner_data_error(0))
        # A repeated old response must not rewind a recovered fallback either.
        self.date_openapi(self.wall - 10)
        await self.fox.publish_cloud_data("SN1")
        self.assertIsNone(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"])

    async def test_openapi_age_expires_even_when_publisher_is_stalled(self):
        """Expiry is measured from the sample time, never the successful request time."""
        self.date_openapi(self.wall - FOX_OPENAPI_MAX_AGE + 1)
        await self.fox.publish_data()
        self.sync_publication()
        self.assertIsNone(self.fox.planner_data_error(0))
        self.wall += 1
        self.assertIn("suspended", self.fox.planner_data_error(0))
        await self.fox.publish_data()
        reading = self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]
        self.assertIsNone(reading["state"])
        self.assertIn("15 minutes", reading["attributes"]["stale_reason"])

    async def test_openapi_unknown_or_future_timestamps_cannot_become_zero(self):
        """A numerical SoC is unusable when its sample time cannot be trusted."""
        for stamp in (None, "bad", datetime.fromtimestamp(self.wall + 61, timezone.utc).isoformat()):
            for item in self.fox.device_values["SN1"].values():
                item["timestamp"] = stamp
            await self.fox.publish_data()
            self.sync_publication()
            self.assertIsNone(self.fox.base.entities["sensor.predbat_fox_sn1_soc"]["state"])
            self.assertIsNotNone(self.fox.planner_data_error(0))

    async def test_equal_sample_and_genuine_soc_increase_remain_valid(self):
        """Chronology checks timestamps, not discharge direction or increasing SoC values."""
        await self.publish_live(soc=23)
        self.date_openapi(self.wall - 5)
        self.stream.values = {}
        self.stream.expires_at = 0
        await self.fox.publish_cloud_data("SN1")
        self.sync_publication()
        self.assertEqual(self.fox.base.entities["sensor.predbat_fox_sn1_soc"]["state"], 42)
        self.assertIsNone(self.fox.planner_data_error(0))

    async def test_aggregate_soc_uses_the_oldest_pack_timestamp(self):
        """A freshly updated pack cannot make another stale pack appear current."""
        self.fox.device_values["SN1"]["SoC_2"] = {"value": 44, "unit": "%", "timestamp": datetime.fromtimestamp(self.wall - FOX_OPENAPI_MAX_AGE, timezone.utc).isoformat()}
        await self.fox.publish_data()
        self.assertIsNone(self.fox.dashboard_items["sensor.predbat_fox_sn1_soc"]["state"])

    async def test_openapi_only_setups_keep_their_existing_age_policy(self):
        """Optional website safety must not alter installations without website credentials."""
        self.fox.cloud_password = None
        self.date_openapi(self.wall - 86400)
        await self.fox.publish_data()
        self.sync_publication()
        reading = self.fox.base.entities["sensor.predbat_fox_sn1_soc"]
        self.assertEqual(reading["state"], 42)
        self.assertNotIn("valid_until", reading["attributes"])
        self.assertIsNone(self.fox.planner_data_error(0))

    async def test_use_time_guard_checks_openapi_sample_time_and_latest_live_sample(self):
        """A delayed HA update cannot bypass ordering or trust a future/missing sample time."""
        await self.fox.publish_data()
        self.sync_publication()
        original = deepcopy(self.fox.base.entities)
        for sampled_at in (None, self.wall + 61):
            self.fox.base.entities = deepcopy(original)
            self.fox.base.entities["sensor.predbat_fox_sn1_soc"]["attributes"]["sample_time"] = sampled_at
            self.assertIsNotNone(self.fox.planner_data_error(0))
        self.fox.base.entities = original
        self.stream.latest_samples["soc"] = self.wall + 1
        self.assertIsNotNone(self.fox.planner_data_error(0))


def run_fox_safety_tests(my_predbat):
    """Run website failure-injection tests through Predbat's normal test harness."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFoxWebsiteSafety)
    return not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
