import time

import pyautogui as pg


class MouseController:
    def __init__(self):
        print("init the mouse position={}".format(pg.position()))

    def click(self, button='left'):
        pg.click(button=button)

    def move(self, position, speed=0.05):
        pg.moveTo(*position, speed)

    def drag(self, position, button='left'):
        pg.dragTo(position[0], position[1], button=button)

    def scroll(self, dis):
        pg.scroll(dis)

    def hscroll(self, dis):
        pg.hscroll(dis)

    def move_and_click(self, position):
        self.move(position)
        self.click()
        self.click()

    def move_and_single_click(self, position):
        self.move(position)
        self.click()
