# Music Genre Classifier And Recommender using Deep Learning

## Process



## Download mp3 files
The first thing I needed to do was download a large number of the sample mp3 files to work with.

The 9 genres were:

Breakbeat
Dancehall
Downtempo
Drum and Bass
Funky House
Hop Hop / R&B
Minimal House
Rock/Indie
Trance


## Convert audio to spectrograms
There’s way too much data contained within an audio file, and so a large part of this whole process is essentially trying to condense the information from the music and extract the main features while eliminating all the ‘noise’. It’s basically an exercise in dimensionality reduction, and the first stage of this was to convert the audio into an image format.

Using Discrete Fourier Transforms to convert the audio signals into the frequency domain, I processed each of my 9,000 mp3 audio files and saved spectrogram images for each song. A spectrogram is a visual representation of the spectrum of frequencies of sound as it varies with time. The intensity of colour on the image represents the amplitude of the sound at that frequency.

I chose to create monochrome spectrograms, like this one below:

![](https://github.com/man007yadav/Music-Classifier-and-Recommender/blob/master/image/spectro.jpg)

This is around 20 seconds of audio generated from a hip hop track. On the x-axis is time, and on the y-axis are the frequencies of the sound.

Split images into 256×256 squares
In order to train a model on this data, I needed all of my images to be of equal dimensions, so I split all of my spectrograms into 256×256 squares. This represents just over 5 seconds of audio on each image.



By now, I had more than 185,000 images in total, each with a label for the music genre it represented.

I split my data into a training set of 120,000, a validation set of 45,000 and a holdout set of 20,000 images.

Train a Convolutional Neural Network on the images
I trained a CNN on my image data. I needed to teach it to recognise what the different types of music ‘looked’ like in the spectrogram images, so I used the genre labels and trained it to identify the music genre from the images.

Below is a visualisation of the CNN pipeline:

![](https://github.com/man007yadav/Music-Classifier-and-Recommender/blob/master/image/cnn_pipeline.jpg)


Starting with the spectrogram image on the upper left hand side, the image is converted into a matrix of numbers representing the colours in each of the pixels. From there, the data passes through various layers in the pipeline and through each layer the shape of the matrix is transformed until it eventually reaches a softmax classifier in the bottom right hand corner. This is a vector of 9 numbers and contains the probabilities for each of the 9 music genres the CNN assigns to the image.

One step in from that is the fully connected layer. This is a vector of 128 numbers and these are essentially 128 music features that have been extracted from the image after passing through the various layers. Another way of thinking about this layer is that all the key information in the original image has been compacted into 128 numbers that ‘explain’ the image.

## So how well did the CNN do?

It was capable of classifying the music genre of a song with 75% accuracy, which I felt was pretty good. Music genres are somewhat subjective and music often transcends more than one genre, so I felt happy that it was doing a good job. Here’s a breakdown of the classification accuracies:

Trance: 91%
Drum & Bass: 90%
Dancehall: 79%
Breakbeat: 78%
Funky House: 71%
Downtempo: 71%
Rock/Indie: 70%
Minimal House: 63%
Hip Hop / R&B: 61%

It did a really good job classifying trance music while at the other end of the scale was hip hop / R&B with 61%, which is still almost 6 times better than randomly assigning a genre to the image. I suspect that there’s some crossover between hip hop, breakbeat and dancehall and that might have resulted in a lower classification accuracy. Trance music is quite different to the other 8 genres in the list, so perhaps that’s also why it did so much better at identifying that type of music.

Nevertheless, these numbers weren’t too important to me; what was important was that it was capable of differentiating between different types of music.

## What about the music recommender?

Now that I had a trained neural network capable of ‘seeing’ music in spectrograms, I no longer needed the softmax classifier, so I removed that layer and extracted the 128 music feature vectors for all 185,000 images in my data set.

With each image representing just over 5 seconds of audio, and the sample mp3 files being around 2 minutes long in total, I had approximately 23 images – and therefore 23 feature vectors – for each music file. I calculated the mean (average) vector for each song, giving me 9,000 feature vectors; one for each of the 9,000 songs I had originally downloaded.


At this point I had now extracted 128 features from the music files that identified different characteristics in the music. So in order to create recommendations of songs that shared similar characteristics, all I needed to find were the vectors that were most similar to one another. To do that, I calculated the cosine similarity between all 9,000 vectors.
