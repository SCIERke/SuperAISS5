# Data Processing and Augmentation

Augmentation help us to create more data for our model. Augmentation have pros and cons, **you must be careful to use it!**

# Image Data Augmentation

- Geometric Transformations:
    - Rotation: Rotating images by various angles.
    - Flipping: Mirroring images horizontally or vertically.
    - Scaling: Resizing images, zooming in or out.
    - Translation: Shifting images horizontally or vertically.
    - Cropping: Extracting random or central portions of images.
    - Shearing: Distorting the shape of images.
- Color Space Transformations:
    - Brightness Adjustment: Altering the overall brightness of images.
    - Contrast Adjustment: Modifying the difference between light and dark areas.
    - Saturation Adjustment: Changing the intensity of colors.
    - Color Jittering: Randomly varying brightness, contrast, and saturation.
- Noise Injection:
    - Gaussian Noise: Adding random noise with a Gaussian distribution.
    - Salt-and-Pepper Noise: Introducing random black and white pixels.
- Kernel Filters:
    - Applying blurring or sharpening filters.
- Random Erasing:
    - Randomly masking out rectangular regions of images.
- Mixup and CutMix:
    - Combining pixels from different images to create new samples.

# Text Data Augmentation

- Synonym Replacement:
    - Replacing words with their synonyms.
- Random Insertion/Deletion:
    - Adding or removing words randomly.
- Word Shuffling:
    - Rearranging the order of words in a sentence.
- Back Translation:
    - Translating text to another language and back.
- Text generation:
    - Using models to create new sentences that have the same meaning

# Audio Data Augmentation

- Noise Injection:
    - Adding background noise or white noise.
- Time Stretching:
    - Speeding up or slowing down audio.
- Pitch Shifting:
    - Changing the pitch of audio signals.
- Time Shifting:
    - Shifting audio signals forward or backward in time.
- Volume Adjustment:
    - Increasing or decreasing the volume of audio.