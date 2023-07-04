import time

import pyautogui as pg


class MouseController:
    def __init__(self):
        print("init the mouse position={}".format(pg.position()))

    def click(self, button='left'):
        pg.click(duration=0.01, button=button)

    def double_click(self, button='left'):
        pg.click(duration=0.01, button=button)
        pg.click(duration=0.01, button=button)

    def move(self, position, speed=0.01):
        pg.moveTo(*position, duration=speed)

    def drag(self, position, button='left'):
        pg.dragTo(position[0], position[1], duration=0.01, button=button)

    def scroll(self, dis):
        pg.scroll(dis)

    def hscroll(self, dis):
        pg.hscroll(dis)

    def vscroll(self, dis):
        pg.vscroll(dis)

    def move_and_click(self, position):
        self.move(position)
        pg.click(duration=0.01)
        pg.click(duration=0.01)

    def move_and_single_click(self, position):
        self.move(position)
        pg.click(duration=0.01)
    
    def get_pixel(self, position):
        return pg.pixel(position[0], position[1])
