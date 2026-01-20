import asyncio
import websockets
import requests
import json
import time

import logging

logger = logging.getLogger(__name__)


class LoginError(Exception):
    pass


class SaveReplayError(Exception):
    pass


class PSWebsocketClient:
    websocket = None
    address = None
    login_uri = None
    username = None
    password = None
    last_message = None
    last_challenge_time = 0

    @classmethod
    async def create(cls, username, password, address, login_uri=None):
        self = PSWebsocketClient()
        self.username = username
        self.password = password
        self.address = address
        self.websocket = await websockets.connect(self.address)
        if login_uri is None:
            self.login_uri = None
        else:
            self.login_uri = login_uri
        return self

    async def join_room(self, room_name):
        message = "/join {}".format(room_name)
        await self.send_message("", [message])
        logger.debug("Joined room '{}'".format(room_name))

    async def receive_message(self):
        message = await self.websocket.recv()
        logger.debug("Received message from websocket: {}".format(message))
        return message

    async def send_message(self, room, message_list):
        message = room + "|" + "|".join(message_list)
        logger.debug("Sending message to websocket: {}".format(message))
        await self.websocket.send(message)
        self.last_message = message

    async def avatar(self, avatar):
        await self.send_message("", ["/avatar {}".format(avatar)])
        await self.send_message("", ["/cmd userdetails {}".format(self.username)])
        while True:
            # Wait for the query response and check the avatar
            # |queryresponse|QUERYTYPE|JSON
            msg = await self.receive_message()
            msg_split = msg.split("|")
            if msg_split[1] == "queryresponse":
                user_details = json.loads(msg_split[3])
                if user_details["avatar"] == avatar:
                    logger.info("Avatar set to {}".format(avatar))
                else:
                    logger.warning(
                        "Could not set avatar to {}, avatar is {}".format(
                            avatar, user_details["avatar"]
                        )
                    )
                break

    async def close(self):
        await self.websocket.close()

    async def get_id_and_challstr(self):
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            if split_message[1] == "challstr":
                return split_message[2], split_message[3]

    async def login(self):
        logger.info("Logging in...")
        if not self.login_uri:
            # Guest login for local servers
            message = ["/trn " + self.username + ",0,"]
            await self.send_message("", message)
            actual_name = await self._wait_for_updateuser(timeout_s=3.0)
            if actual_name:
                self.username = actual_name
            return self._to_id(self.username)
        client_id, challstr = await self.get_id_and_challstr()
        response = requests.post(
            self.login_uri,
            data={
                "name": self.username,
                "pass": self.password,
                "challstr": "|".join([client_id, challstr]),
            },
        )

        if response.status_code != 200:
            logger.error("Could not log-in\nDetails:\n{}".format(response.content))
            raise LoginError("Could not log-in")

        response_json = json.loads(response.text[1:])
        if "actionsuccess" not in response_json:
            logger.error("Login Unsuccessful: {}".format(response_json))
            raise LoginError("Could not log-in: {}".format(response_json))

        assertion = response_json.get("assertion")
        message = ["/trn " + self.username + ",0," + assertion]
        logger.info("Successfully logged in")
        await self.send_message("", message)
        actual_name = await self._wait_for_updateuser(timeout_s=5.0)
        if actual_name:
            self.username = actual_name
        return self._to_id(self.username)

    def _to_id(self, name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    async def _wait_for_updateuser(self, timeout_s: float = 3.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self.receive_message(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            split_msg = msg.split("|")
            if len(split_msg) > 2 and split_msg[1] == "updateuser":
                return split_msg[2]
        return None

    async def update_team(self, team):
        await self.send_message("", ["/utm {}".format(team)])

    async def challenge_user(self, user_to_challenge, battle_format):
        logger.info("Challenging {}...".format(user_to_challenge))
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        await self.send_message("", message)
        self.last_challenge_time = time.time()

    async def accept_challenge(self, battle_format, room_name):
        if room_name is not None:
            await self.join_room(room_name)

        logger.info("Waiting for a {} challenge".format(battle_format))
        username = None
        while username is None:
            msg = await self.receive_message()
            split_msg = msg.split("|")
            if len(split_msg) < 2:
                continue
            if split_msg[1] == "updatechallenges":
                try:
                    payload = json.loads(split_msg[2])
                    challenges = payload.get("challengesFrom", {})
                    for challenger, fmt in challenges.items():
                        if fmt == battle_format:
                            logger.info(
                                "Challenge received via updatechallenges from %s", challenger
                            )
                            username = challenger
                            break
                except Exception:
                    pass
                if username is None:
                    continue
            if len(split_msg) < 5 or split_msg[1] != "pm":
                continue
            sender = split_msg[2].strip()
            receiver = split_msg[3].strip().replace("!", "").replace("‽", "")
            if self._to_id(receiver) != self._to_id(self.username):
                continue
            message = split_msg[4].strip()
            if not message.startswith("/challenge"):
                continue
            fmt = None
            if "," in message:
                fmt = message.split(",")[-1].strip()
            if fmt and fmt != battle_format:
                continue
            logger.info("Challenge received via PM from %s", sender)
            username = sender

        message = ["/accept " + username]
        logger.info("Accepting challenge from %s", username)
        await self.send_message("", message)

    async def search_for_match(self, battle_format):
        logger.info("Searching for ranked {} match".format(battle_format))
        message = ["/search {}".format(battle_format)]
        await self.send_message("", message)

    async def leave_battle(self, battle_tag):
        message = ["/leave {}".format(battle_tag)]
        await self.send_message("", message)

        while True:
            msg = await self.receive_message()
            if battle_tag in msg and "deinit" in msg:
                return

    async def save_replay(self, battle_tag):
        message = ["/savereplay"]
        await self.send_message(battle_tag, message)
