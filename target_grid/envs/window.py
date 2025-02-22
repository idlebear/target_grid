import numpy as np
import pygame
from math import sqrt
from util import DotDict

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_MARGIN = 120
DEFAULT_DISPLAY_SIZE = 20


Colours = DotDict(
    {
        "black": (0, 0, 0, 255),
        "white": (255, 255, 255, 255),
        "grey": (128, 128, 128, 255),
        "light_grey": (211, 211, 211, 255),
        "dark_grey": (169, 169, 169, 255),
        "red": (255, 0, 0, 255),
        "green": (0, 255, 0, 255),
        "blue": (0, 0, 255, 255),
        "yellow": (255, 255, 0, 255),
        "cyan": (0, 255, 255, 255),
        "magenta": (255, 0, 255, 255),
        "orange": (255, 165, 0, 255),
        "purple": (128, 0, 128, 255),
        "brown": (165, 42, 42, 255),
        "pink": (255, 192, 203, 255),
        "light_blue": (173, 216, 230, 255),
    }
)

SCREEN_OUTLINE_COLOUR = Colours.black
SCREEN_BACKGROUND_COLOUR = Colours.white
STATUS_FONT_COLOUR = Colours.blue
STATUS_FONT_SIZE = 32


STATUS_Y_SIZE = 100
STATUS_YMARGIN = 8
STATUS_X_SIZE = 300
STATUS_XMARGIN = 16


def get_location(origin, location):
    return [location[0] - origin[0], 1 - (location[1] - origin[1])]


class Window:
    def __init__(
        self,
        screen_width,
        screen_height,
        margin,
        display_origin=(0, 0),
        display_size=10.0,
        frame_rate=4,
    ):

        pygame.font.init()
        self.sim_time_text = pygame.font.SysFont("dejavuserif", 15)
        self.elapsed_time_text = pygame.font.SysFont("dejavuserif", 10)
        self.status_font = pygame.font.SysFont("roboto", STATUS_FONT_SIZE)

        self.xmargin = margin * 0.5
        self.ymargin = margin * 0.5
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.env_size = min(screen_height, screen_width) - margin
        self.scale = self.env_size / display_size  # pixels per meter
        self.display_size = display_size
        self.border_offset = 10
        self.origin = display_origin

        self.screen = pygame.Surface(
            (self.screen_width, self.screen_height), flags=pygame.SRCALPHA
        )
        self.tmp_screen = pygame.Surface(
            (self.screen_width, self.screen_height), flags=pygame.SRCALPHA
        )
        self.display_screen = None
        self.clock = None
        self.frame_rate = frame_rate

    # def _get_location_on_screen(self, origin, location):
    #     return [
    #         int(self.xmargin + (location[0]-origin[0] - EGO_X_OFFSET)*self.env_size),
    #         int(self.ymargin + (location[1]-origin[1] - EGO_Y_OFFSET)*self.env_size)
    #     ]

    @staticmethod
    def initialize_screen():
        pygame.init()
        pygame.display.init()

    def clear(self):
        self.screen.fill(SCREEN_BACKGROUND_COLOUR)

    def get_drawing_scale(self):
        return self.scale

    def draw_polyline(self, points, colour, width=2):
        start = points[0]
        for end in points[1:]:
            self.draw_line(start, end, colour, width)
            start = end

    def draw_line(self, start, end, colour, width=2, use_transparency=False):
        sx = self.xmargin + int((start[0] - self.origin[0]) * self.scale)
        sy = self.ymargin + int((start[1] - self.origin[1]) * self.scale)
        ex = self.xmargin + int((end[0] - self.origin[0]) * self.scale)
        ey = self.ymargin + int((end[1] - self.origin[1]) * self.scale)

        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            draw_screen = self.tmp_screen
        else:
            draw_screen = self.screen

        pygame.draw.line(
            draw_screen, color=colour, start_pos=(sx, sy), end_pos=(ex, ey), width=width
        )

        if use_transparency:
            self.screen.blit(self.tmp_screen, (0, 0))

    def draw_circle(self, center, colour, radius=2, use_transparency=False):
        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            draw_screen = self.tmp_screen
        else:
            draw_screen = self.screen

        cx = self.xmargin + int((center[0] - self.origin[0]) * self.scale)
        cy = self.ymargin + int((center[1] - self.origin[1]) * self.scale)
        radius = int(radius * self.scale)

        pygame.draw.circle(draw_screen, color=colour, center=(cx, cy), radius=radius)
        if use_transparency:
            self.screen.blit(self.tmp_screen, (0, 0))

    def draw_rect(
        self,
        center,
        height,
        width=None,
        colour=Colours.pink,
        border_width=0,
        border_colour=None,
        use_transparency=False,
    ):
        if width is None:
            width = height

        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            draw_screen = self.tmp_screen
        else:
            draw_screen = self.screen

        pygame.draw.rect(
            draw_screen,
            color=colour,
            rect=(
                self.xmargin
                + int((center[0] - width / 2.0 - self.origin[0]) * self.scale),
                self.ymargin
                + int((center[1] - height / 2.0 - self.origin[1]) * self.scale),
                int(width * self.scale),
                int(height * self.scale),
            ),
        )

        if border_width > 0:
            if border_colour is None:
                border_colour = Colours.black
            pygame.draw.rect(
                draw_screen,
                color=border_colour,
                rect=(
                    self.xmargin
                    + int((center[0] - width / 2.0 - self.origin[0]) * self.scale),
                    self.ymargin
                    + int((center[1] - height / 2.0 - self.origin[1]) * self.scale),
                    int(width * self.scale),
                    int(height * self.scale),
                ),
                width=border_width,
            )

        if use_transparency:
            self.screen.blit(self.tmp_screen, (0, 0))

    def draw_triangle(
        self,
        center,
        size,
        orientation,
        colour,
        border_width=0,
        border_colour=None,
        use_transparency=False,
    ):
        # points = [
        #     (0, size / sqrt(3)),
        #     (-size / 2, -size / (2 * sqrt(3))),
        #     (size / 2, -size / (2 * sqrt(3))),
        # ]
        # a pointier triangle
        points = [
            (size / sqrt(3), 0),
            (
                -size / (2 * sqrt(3)),
                -size / 3,
            ),
            (-size / (2 * sqrt(3)), size / 3),
        ]
        # rotate the points
        points = [
            (
                x * np.cos(orientation) - y * np.sin(orientation),
                x * np.sin(orientation) + y * np.cos(orientation),
            )
            for x, y in points
        ]
        points = [(x + center[0], y + center[1]) for x, y in points]
        self.draw_polygon(
            fill_colour=colour,
            points=points,
            border_width=border_width,
            border_colour=border_colour,
            use_transparency=use_transparency,
        )

    def draw_polygon(
        self,
        fill_colour,
        points,
        border_width=2,
        border_colour=Colours.black,
        use_transparency=False,
    ):
        points = [
            [
                self.xmargin + int((x - self.origin[0]) * self.scale),
                self.ymargin + int((y - self.origin[1]) * self.scale),
            ]
            for x, y in points
        ]

        if border_colour is None:
            border_colour = Colours.black
        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            if fill_colour is not None:
                pygame.draw.polygon(self.tmp_screen, fill_colour, points, 0)
            if border_width > 0:
                pygame.draw.polygon(
                    self.tmp_screen, border_colour, points, width=border_width
                )
            self.screen.blit(self.tmp_screen, (0, 0))
        else:
            if fill_colour is not None:
                pygame.draw.polygon(self.screen, fill_colour, points, 0)
            if border_width > 0:
                pygame.draw.polygon(
                    self.screen, border_colour, points, width=border_width
                )

    # Quick image rotation
    #   https://stackoverflow.com/questions/4183208/how-do-i-rotate-an-image-around-its-center-using-pygame
    def draw_image(self, image, center, orientation):
        center = (
            self.xmargin + int((center[0] - self.origin[0]) * self.scale),
            self.ymargin + int((center[1] - self.origin[1]) * self.scale),
        )
        rotated_image = pygame.transform.rotate(image, np.rad2deg(orientation))
        new_rect = rotated_image.get_rect(center=image.get_rect(center=center).center)
        self.screen.blit(rotated_image, new_rect)

    def draw_status(self, collisions, sim_time):
        #  draw the limits of the environment
        pygame.draw.rect(
            self.screen,
            SCREEN_OUTLINE_COLOUR,
            (
                self.xmargin - self.border_offset,
                self.ymargin - self.border_offset,
                self.env_size + self.border_offset * 2,
                self.env_size + self.border_offset * 2,
            ),
            2,
        )

        collisions_str = f"Collisions: {collisions}"
        time_str = f"Sim Time: {sim_time:.4f}"

        text_width, text_height = self.status_font.size(collisions_str)
        time_text_width, time_text_height = self.status_font.size(time_str)

        x_avg_offset = self.env_size + self.xmargin - text_width - STATUS_XMARGIN * 2
        y_avg_offset = self.env_size + self.ymargin - text_height - STATUS_YMARGIN

        pygame.draw.rect(
            self.screen,
            SCREEN_BACKGROUND_COLOUR,
            (
                x_avg_offset - STATUS_XMARGIN,
                y_avg_offset - STATUS_YMARGIN,
                text_width + STATUS_XMARGIN * 2,
                text_height + STATUS_YMARGIN,
            ),
            0,
        )
        pygame.draw.rect(
            self.screen,
            SCREEN_OUTLINE_COLOUR,
            (
                x_avg_offset - STATUS_XMARGIN,
                y_avg_offset - STATUS_YMARGIN,
                text_width + STATUS_XMARGIN * 2,
                text_height + STATUS_YMARGIN,
            ),
            2,
        )
        text = self.status_font.render(collisions_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (x_avg_offset + STATUS_XMARGIN / 3, y_avg_offset))

        pygame.draw.rect(
            self.screen,
            SCREEN_BACKGROUND_COLOUR,
            (
                self.xmargin + STATUS_XMARGIN / 2,
                self.xmargin + STATUS_YMARGIN,
                time_text_width + STATUS_XMARGIN,
                time_text_height + STATUS_YMARGIN,
            ),
            0,
        )

        text = self.status_font.render(time_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(
            text, (self.xmargin + STATUS_XMARGIN, self.ymargin + STATUS_YMARGIN)
        )

    def display(self):
        if self.display_screen is None:
            self.display_screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Target Grid")
        self.display_screen.blit(self.screen, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to
        # keep the framerate stable.
        self.clock.tick(self.frame_rate)

    def render(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def save_frame(self, path):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.display.quit()
        pygame.quit()
