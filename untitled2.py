#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:57:56 2023

@author: evgeni
"""

from PIL import Image, ImageDraw
import imageio

# Set the dimensions of the image and the number of frames
image_width = 200
image_height = 200
num_frames = 10

# Create a list to store the frames
frames = []

# Generate frames for the animation
for i in range(num_frames):
    # Create a new image with a white background
    image = Image.new("RGB", (image_width, image_height), "black")
    
    # Draw a circle at a different position in each frame
    draw = ImageDraw.Draw(image)
    radius = 20
    center_x = (i + 5) * (image_width // (num_frames + 1))
    center_y = image_height // 2
 #   draw.ellipse(shape, fill ="# 800080", outline ="green")
    draw.ellipse([(center_x - radius+i/10, center_y - radius+i/10), (center_x + radius, center_y + radius)], width=4, fill="black", outline="red")
    
    # Add the image to the list of frames
    frames.append(image)

# Save the frames as a GIF animation
imageio.mimsave("animation.gif", frames, duration=0.5)
