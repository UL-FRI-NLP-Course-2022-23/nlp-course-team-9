#!/bin/python3

import os
import requests
import sys


def send_status(info_msg):
    """
    Upon job submission, a message is sent to the Telegram channel specified in
    tg_credentials.txt file.
    """
    credentials_loc = "slurm/tg_credentials.txt"

    if os.path.isfile(credentials_loc):
        with open(credentials_loc) as credentials_file:
            bot_token, channel_token = credentials_file.readline().split()
    
        with requests.session() as s:
            a = s.post("https://api.telegram.org/bot" + bot_token + "/sendMessage?chat_id=-100" + channel_token[1:] + "&text=" + info_msg)
            return True


if __name__ == "__main__":
    send_status(sys.argv[1])
