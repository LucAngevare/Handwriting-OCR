import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('handwriting_model.h5')
image = cv2.imread('woord.png', cv2.IMREAD_GRAYSCALE)
image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
predicted_characters = []
character_positions = []

for label in range(1, num_labels):
    left = stats[label, cv2.CC_STAT_LEFT]
    top = stats[label, cv2.CC_STAT_TOP]
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    if width > 0 and height > 0:
        component = image[top:top+height, left:left+width]
        desired_height = 28
        aspect_ratio = float(width) / float(height)
        desired_width = int(aspect_ratio * desired_height)

        if desired_width > 28:
            desired_width = 28
            desired_height = int(desired_width / aspect_ratio)

        resized_component = cv2.resize(component, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
        padded_component = np.ones((desired_height, 28), dtype=np.uint8) * 255
        pad_left = (28 - desired_width) // 2
        pad_right = 28 - desired_width - pad_left
        padded_component[:, pad_left:pad_left+desired_width] = resized_component
        normalized_component = cv2.resize(padded_component, (28, 28), interpolation=cv2.INTER_AREA).reshape(1, 28, 28, 1) / 255.0
        predictions = model.predict(normalized_component)
        predicted_character = chr(np.argmax(predictions) + 65)
        predicted_characters.append(predicted_character)
        character_positions.append((left, top))

sorted_characters = [char for _, char in sorted(zip(character_positions, predicted_characters))]
predicted_text = "".join(sorted_characters)

print("Predicted Text:", predicted_text)
