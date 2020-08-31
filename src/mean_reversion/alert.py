import discord


class Messenger(discord.Client):
    message = "Default"

    def __init__(self, message):
        super().__init__()
        self.message = message

    async def on_ready(self):
        print('Logged in as')
        print('    UN: ' + self.user.name)
        print('    ID: ' + str(self.user.id))
        await self.send_message(self.message)
        print('Message sent:\n' + '    ' + self.message)

    async def send_message(self, msg: str):
        channel = self.get_channel(707352560557883392)
        await channel.send(msg)


def send_message(msg: str):
    client = Messenger(msg)
    client.run('NzA1OTMzOTk2NTA4NDQ2NzMw.XrHuiA.wEIjUXEc8rZso6IOIB1a2DG3eco')
