import pygame
import time

pygame.mixer.init()
sound = pygame.mixer.Sound("Audio/wrong_5.mp3")
pygame.mixer.stop()
sound.play()
print("Playing sound...")
time.sleep(2)  # wait for sound to finish
