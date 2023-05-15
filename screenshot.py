from PIL import ImageGrab
import os


class ScreenShot:
    def __init__(self, position, path):
        self.position = position
        self.path = path

    def run(self):
        im = ImageGrab.grab(bbox=self.position)
        im.save(os.path.join(self.path, "ss.png"))


# ss = ScreenShot((0, 37, 413, 782+37), "../samples/")
# ss.run("ss")
