import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

def create_initial_state():
    head = Point(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    snake = [head,
             Point(head.x - BLOCK_SIZE, head.y),
             Point(head.x - (2 * BLOCK_SIZE), head.y)]
    return {
        'direction': Direction.RIGHT,
        'snake': snake,
        'score': 0,
        'food': place_food(snake),
        'frame_iteration': 0
    }

def place_food(snake):
    while True:
        x = random.randint(0, (WINDOW_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (WINDOW_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food = Point(x, y)
        if food not in snake:
            return food

def move(state, action):
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(state['direction'])

    if np.array_equal(action, [1, 0, 0]):
        new_dir = clock_wise[idx]
    elif np.array_equal(action, [0, 1, 0]):
        new_dir = clock_wise[(idx + 1) % 4]
    else: # [0, 0, 1]
        new_dir = clock_wise[(idx - 1) % 4]

    head = state['snake'][0]
    if new_dir == Direction.RIGHT:
        new_head = Point(head.x + BLOCK_SIZE, head.y)
    elif new_dir == Direction.LEFT:
        new_head = Point(head.x - BLOCK_SIZE, head.y)
    elif new_dir == Direction.DOWN:
        new_head = Point(head.x, head.y + BLOCK_SIZE)
    elif new_dir == Direction.UP:
        new_head = Point(head.x, head.y - BLOCK_SIZE)

    return new_dir, new_head

def is_collision(pt, snake):
    return (pt.x > WINDOW_WIDTH - BLOCK_SIZE or pt.x < 0 or
            pt.y > WINDOW_HEIGHT - BLOCK_SIZE or pt.y < 0 or
            pt in snake[1:])

def play_step(state, action):
    new_state = state.copy()
    new_state['frame_iteration'] += 1

    new_direction, new_head = move(new_state, action)
    new_state['direction'] = new_direction
    new_snake = [new_head] + new_state['snake'][:-1]

    reward = 0
    game_over = False

    if is_collision(new_head, new_state['snake']) or new_state['frame_iteration'] > 100 * len(new_state['snake']):
        game_over = True
        reward = -10
        return new_state, reward, game_over

    if new_head == new_state['food']:
        new_state['score'] += 1
        reward = 10
        new_snake = [new_head] + new_state['snake']
        new_state['food'] = place_food(new_snake)

    new_state['snake'] = new_snake
    return new_state, reward, game_over

def update_ui(display, state):
    display.fill(BLACK)

    for pt in state['snake']:
        pygame.draw.rect(display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

    pygame.draw.rect(display, RED, pygame.Rect(state['food'].x, state['food'].y, BLOCK_SIZE, BLOCK_SIZE))

    font = pygame.font.SysFont(None, 25)
    text = font.render("Score: " + str(state['score']), True, WHITE)
    display.blit(text, [0, 0])
    pygame.display.flip()

def run_game():
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()

    state = create_initial_state()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Here you would get the action from your AI model
        action = [1, 0, 0]  # placeholder

        state, reward, game_over = play_step(state, action)

        update_ui(display, state)
        clock.tick(SPEED)

        if game_over:
            state = create_initial_state()

if __name__ == "__main__":
    run_game()
