# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------

"""Fox website read-only status, authentication, recovery and source selection.

This module owns the optional website protocol. The Fox component supplies
credentials, logging and publication; inverter commands remain in fox.py.
"""

import asyncio
from datetime import datetime
import hashlib
import math
import re
import time
from urllib.parse import urlencode

import aiohttp
import pytz

FOX_DOMAIN = "https://www.foxesscloud.com"
FOX_LANG = "en"

# The website's live energy-flow socket is separate from OpenAPI telemetry. Never
# promote its cached/keepalive frames to current readings, even while connected.
FOX_CLOUD_MAX_AGE = 30
FOX_CLOUD_RECOVERY_SECONDS = 120
FOX_CLOUD_LOGIN_PATH = "/basic/v0/user/login"
FOX_CLOUD_WS_PATH = "/dew/v0/wsmaitian"


def fox_cloud_headers(path, timestamp, local_tz):
    """Sign a website login request using the v2 portal's signature format.

    Verified against /v2/js/signature.wasm (2026-11-03): literal backslash-r/n
    separators, language before timestamp, and the fixed .5245784 suffix. Keep
    this separate from the OpenAPI signature and its API key/OAuth credentials.
    """
    signature = hashlib.md5(rf"{path}\r\n\r\n{FOX_LANG}\r\n{timestamp}".encode("utf-8")).hexdigest() + ".5245784"
    now = datetime.fromtimestamp(int(timestamp) / 1000, local_tz).strftime("%Y-%m-%d %H:%M:%S")
    headers = {
        "Content-Type": "application/json",
        "lang": FOX_LANG,
        "token": "",
        "timestamp": str(timestamp),
        "signature": signature,
        "timezone": str(local_tz),
        "dt": f"{local_tz}@{timestamp}@{now}",
        "platform": "web",
    }
    headers["X-Auth-Platform"] = "FOX_CLOUD_WEB_2_0"
    return headers


def fox_cloud_number(value):
    """Return a finite telemetry number, rejecting booleans and placeholders."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def fox_sample_time(value, local_tz):
    """Parse Fox's inverter-local sample time, including BST/GMT numeric offsets."""
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        return None
    if not re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", value.strip()):
        return None
    # The documented format ends in zZ, e.g. BST+0100. Trust the numeric
    # offset rather than platform-specific timezone abbreviation parsing.
    value = re.sub(r" [A-Za-z]{1,8}(?=[+-]\d{4}$)", " ", value.strip())
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = local_tz.localize(parsed, is_dst=None)
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError, pytz.InvalidTimeError):
        return None


def fox_latest_status_values(openapi, website, live_variables, local_tz, now):
    """Choose the newest valid status per field, retaining v2 samples through live gaps."""
    values = openapi.copy()
    names = {key.casefold(): key for key in values}
    # A newer API net reading also supplies missing directional flow sensors.
    for net, positive, negative in (("invbatpower", "batDischargePower", "batChargePower"), ("meterpower", "gridConsumptionPower", "feedinPower")):
        item = values.get(names.get(net), {})
        power = fox_cloud_power({"power": item})
        if power is not None:
            for variable, sign in ((positive, 1), (negative, -1)):
                if variable.casefold() not in names:
                    values[variable] = {**item, "value": max(sign * power, 0), "unit": "kW", "name": variable}
                    names[variable.casefold()] = variable
    for variable, item in website.items():
        key = variable.casefold()
        original_key = names.get(key)
        original = values.get(original_key, {})
        api_time = fox_sample_time(original.get("timestamp"), local_tz)
        number = fox_cloud_number(original.get("value"))
        valid = number is not None and original.get("unit") in (("%",) if key == "soc" else ("W", "kW"))
        if valid:
            valid = 0 <= number <= 100 if key == "soc" else math.isfinite(number * 1000) and (key in ("invbatpower", "meterpower") or number >= 0)
        if valid and api_time is not None and item["sample_time"] < api_time <= now + 60:
            item, sampled_at = original, api_time
        else:
            cached = key not in live_variables
            item = {**item, "source": "fox_cloud_cached" if cached else "fox_cloud", "cached": cached}
            sampled_at = item["sample_time"]
        values[original_key or variable] = {**item, "sample_time": sampled_at, "sample_age_seconds": max(0, now - sampled_at)}
    return values


def fox_cloud_power(node):
    """Read a visible flow node's power in kW, respecting its supplied unit."""
    if not isinstance(node, dict) or node.get("display") is False:
        return None
    power = node.get("power")
    if not isinstance(power, dict) or power.get("display") is False:
        return None
    value = fox_cloud_number(power.get("value"))
    unit = power.get("unit")
    if value is None or unit not in ("W", "kW") or not math.isfinite(value * (1000 if unit == "kW" else 1)):
        return None
    return value / 1000.0 if unit == "W" else value


def fox_cloud_values(message):
    """Map fresh website flow readings to existing Fox sensor variable names.

    Energy counters and detailed telemetry are deliberately left to OpenAPI:
    the five-second socket does not carry those readings. The feed is per plant,
    so callers must only bind it to a single inverter in that plant.
    """
    if not isinstance(message, dict) or message.get("errno") != 0:
        return {}, None
    result = message.get("result")
    if not isinstance(result, dict):
        return {}, None
    age = fox_cloud_number(result.get("timeDiff"))
    node = result.get("node")
    if age is None or not 0 <= age < FOX_CLOUD_MAX_AGE or not isinstance(node, dict):
        return {}, None
    values = {}
    bat = node.get("bat")
    if isinstance(bat, dict) and bat.get("display") is not False:
        soc = fox_cloud_number(bat.get("soc"))
        if soc is not None and 0 <= soc <= 100 and bat.get("socDisplay") is not False:
            values["SoC"] = {"value": soc, "name": "SoC", "unit": "%"}
        power = fox_cloud_power(bat)
        direction = str(bat.get("charge", ""))
        if power is not None and power >= 0 and direction in ("1", "2"):
            values["batChargePower"] = {"value": power if direction == "1" else 0.0, "name": "Charge Power", "unit": "kW"}
            values["batDischargePower"] = {"value": power if direction == "2" else 0.0, "name": "Discharge Power", "unit": "kW"}
            values["invBatPower"] = {"value": power if direction == "2" else -power, "name": "Battery Power", "unit": "kW"}
    for key, variable, name in [("solar", "pvPower", "PV Power"), ("load", "loadsPower", "Load Power")]:
        power = fox_cloud_power(node.get(key))
        if power is not None and power >= 0:
            values[variable] = {"value": power, "name": name, "unit": "kW"}
    grid = node.get("grid")
    power = fox_cloud_power(grid)
    # The v2 website defines 1=exporting, 2=importing, 3=on-grid, 4=off-grid.
    # An unknown direction must not turn into a fabricated import/export value.
    direction = str(grid.get("gridStatus", "")) if isinstance(grid, dict) else ""
    if power is not None and power >= 0 and direction in ("1", "2"):
        values["gridConsumptionPower"] = {"value": power if direction == "2" else 0.0, "name": "Grid Consumption Power", "unit": "kW"}
        values["feedinPower"] = {"value": power if direction == "1" else 0.0, "name": "Feed-in Power", "unit": "kW"}
        values["meterPower"] = {"value": power if direction == "2" else -power, "name": "Grid Power", "unit": "kW"}
    for item in values.values():
        item.update({"timestamp": result.get("lastUpdateDate", ""), "source": "fox_cloud"})
    return values, age


class FoxCloudError(ValueError):
    """A website failure containing only a controlled, credential-free reason."""


def fox_cloud_failure(error):
    """Describe a failure without exposing server messages or token-bearing URLs."""
    if isinstance(error, FoxCloudError):
        return str(error)
    if isinstance(error, aiohttp.ClientResponseError):
        return f"HTTP {error.status}"
    if isinstance(error, asyncio.TimeoutError):
        return "request timed out or status stream stopped updating"
    return type(error).__name__


class FoxCloudStream:
    """Optional read-only website stream, with bounded freshness and reconnects."""

    def __init__(self, owner, device_sn, plant_id):
        """Keep this stream's live values separate from the OpenAPI cache."""
        self.owner = owner
        self.device_sn = device_sn
        self.plant_id = plant_id
        self.values = {}
        self.expires_at = 0
        self.task = None
        self.received_data = False

    def current_values(self):
        """Return only readings that are still fresh according to inverter age."""
        return {key: item for key, item in self.values.items() if item["valid_until"] > time.time()} if time.monotonic() < self.expires_at else {}

    async def login(self, session):
        """Use website credentials to obtain a token without touching OpenAPI auth."""
        owner = self.owner
        timestamp = str(int(time.time() * 1000))
        path = FOX_CLOUD_LOGIN_PATH
        headers = fox_cloud_headers(path, timestamp, owner.local_tz)
        body = {"user": owner.cloud_username, "password": hashlib.md5(owner.cloud_password.encode("utf-8")).hexdigest(), "type": 1, "verification": 1}
        async with session.post(FOX_DOMAIN + path, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=20)) as response:
            response.raise_for_status()
            data = await response.json()
        result = data.get("result") if isinstance(data, dict) else None
        token = result.get("token") if isinstance(result, dict) else None
        if not isinstance(data, dict) or data.get("errno") != 0 or not isinstance(token, str) or not token:
            code = data.get("errno") if isinstance(data, dict) else None
            reason = f"login rejected (errno={code})" if isinstance(code, int) and code else "login response missing token"
            raise FoxCloudError(reason)
        return token

    async def accept_values(self, values, age):
        """Publish a source transition once and retain its bounded status snapshot."""
        sampled_at = time.time() - age
        for item in values.values():
            item.update(source="fox_cloud", website_version="v2", sample_age_known=True, sample_time=sampled_at, sample_time_estimated=True, valid_until=sampled_at + FOX_CLOUD_MAX_AGE)
        self.values.update(values)
        self.received_data = True
        if not self.expires_at:
            self.owner.log(f"Fox: v2 website status active for {self.device_sn} (live flow)")
        self.expires_at = time.monotonic() + FOX_CLOUD_MAX_AGE - age
        await self.owner.publish_cloud_data(self.device_sn)

    async def listen(self, session, token):
        """Receive fresh flow frames; cached frames cannot extend the deadline."""
        query = urlencode({"plantId": self.plant_id, "token": token, "platform": "web", "lang": FOX_LANG})
        url = FOX_DOMAIN.replace("https://", "wss://") + FOX_CLOUD_WS_PATH + "?" + query
        async with session.ws_connect(url, timeout=aiohttp.ClientWSTimeout(ws_close=5), max_msg_size=1024 * 1024) as socket:
            await socket.send_str("getdata")
            deadline = time.monotonic() + FOX_CLOUD_MAX_AGE
            recovery_deadline = deadline
            # Match the portal's idle requests while keeping data expiry
            # separate: the browser can recover on this socket after a gap.
            next_request = time.monotonic() + 30
            requests_sent = 1
            last_age = last_errno = None
            while True:
                now = time.monotonic()
                if now >= recovery_deadline:
                    self.owner.log(f"Warn: Fox: v2 website freshness deadline reached for {self.device_sn} (last frame errno={last_errno}, timeDiff={last_age}s, getdata requests={requests_sent})")
                    raise asyncio.TimeoutError("No fresh website readings")
                if now >= deadline and self.expires_at:
                    self.expires_at = 0
                    self.owner.log(f"Warn: Fox: v2 website readings expired for {self.device_sn}; using freshest known status while retaining socket for recovery")
                    await self.owner.publish_cloud_data(self.device_sn, refresh=True)
                    deadline = float("inf")
                    continue
                if now >= next_request:
                    await socket.send_str("getdata")
                    requests_sent += 1
                    next_request = now + 30
                wake_deadline = min(recovery_deadline, next_request, deadline)
                try:
                    message = await asyncio.wait_for(socket.receive(), timeout=max(0, wake_deadline - now))
                except asyncio.TimeoutError:
                    continue
                if time.monotonic() >= wake_deadline:
                    # A callback that runs after its scheduled wake could have
                    # been queued during an event-loop stall. Do not re-age it.
                    raise asyncio.TimeoutError("Website frame arrived after its receive deadline")
                if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                    raise ConnectionError("Website stream closed")
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = message.json()
                except (ValueError, TypeError):
                    continue
                next_request = time.monotonic() + 30
                last_errno = fox_cloud_number(data.get("errno")) if isinstance(data, dict) else None
                result = data.get("result") if isinstance(data, dict) else None
                last_age = fox_cloud_number(result.get("timeDiff")) if isinstance(result, dict) else None
                values, age = fox_cloud_values(data)
                if not values:
                    continue
                received_at = time.monotonic()
                await self.accept_values(values, age)
                deadline = self.expires_at
                recovery_deadline = received_at + FOX_CLOUD_RECOVERY_SECONDS

    async def run(self):
        """Run one v2 session/retry loop, retaining cached status between connections."""
        token, delay = None, 5
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            while True:
                self.received_data = False
                try:
                    if token is None:
                        token = await self.login(session)
                    await self.listen(session, token)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if self.received_data:
                        delay = 5
                    else:
                        token = None
                    self.owner.log(f"Warn: Fox: v2 website failed for {self.device_sn} ({fox_cloud_failure(error)}); retaining freshest known status, retry in {delay}s")
                finally:
                    had_live_data = bool(self.expires_at)
                    self.expires_at = 0
                    await self.owner.publish_cloud_data(self.device_sn, refresh=had_live_data)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)
