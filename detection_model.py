import tensorflow as tf
import numpy as np
import cv2

class VideoClassifier:
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['compressed', 'extended']
    
    def classify_batch(self, frames):
        """
        Classify a batch of frames using the loaded model.
        :param frames: Batch of preprocessed frames.
        :return: List of tuples (class_name, probability)
        """
        preds = self.model.predict(frames)  # Predict probabilities for each frame
        results = []

        for pred in preds:
            if len(pred) == 1:  # Binary classification (single output)
                prob_A = pred[0]  # Probability of class 'compressed'
                prob_B = 1 - prob_A  # Probability of class 'extended'
            elif len(pred) == 2:  # Multi-class classification (two outputs)
                prob_A = pred[0]  # Probability of class 'compressed'
                prob_B = pred[1]  # Probability of class 'extended'
            else:
                raise ValueError("Model output shape is not supported.")
            
            # If the highest probability is below a certain threshold, make it uncertain
            # if abs(prob_A - prob_B) < 0.2:
            #     results.append(("uncertain", max(prob_A, prob_B)))
            if prob_A > prob_B:
                results.append((self.class_names[0], prob_A))
            else:
                results.append((self.class_names[1], prob_B))
        
        return results

    def process_video(self, video_path, sample_rate=5, batch_size=16):
        """
        Process the video file and classify frames in batches.
        :param video_path: Path to the video file.
        :param sample_rate: Sample every nth frame to reduce processing.
        :param batch_size: Number of frames to process in a single batch.
        :return: Final prediction and adjusted maximum probability.
        """
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
                frames.append(frame / 255.0)  # Normalize the frame
            frame_count += 1

        video_capture.release()

        frames_array = np.array(frames)
        predictions = []
        for i in range(0, len(frames_array), batch_size):
            batch = frames_array[i:i + batch_size]
            batch_predictions = self.classify_batch(batch)
            predictions.extend(batch_predictions)
        
        # Get the class with most votes and maximum probabilities
        class_votes = {}
        max_probabilities = {}
        for class_name, prob in predictions:
            class_votes[class_name] = class_votes.get(class_name, 0) + 1
            if class_name not in max_probabilities or prob > max_probabilities[class_name]:
                max_probabilities[class_name] = prob

        final_class = max(class_votes, key=class_votes.get)

        # If the final class is "uncertain", directly return 50% probability
        if final_class == "uncertain":
            return final_class, 20.00

        # Otherwise, apply the original logic for "compressed" or "extended"
        prob_A = max_probabilities.get(self.class_names[0], 0)
        prob_B = max_probabilities.get(self.class_names[1], 0)
        max_probability = max(prob_A, prob_B) - abs(prob_A - prob_B)

        # Ensure max_probability is within [0, 1]
        max_probability = min(max(max_probability, 0), 1)

        # If max_probability is greater than or equal to 0.8, set it to 0.5467
        if max_probability >= 0.8:
            max_probability = 0.5467

        # Round the max_probability to two decimal places before returning
        max_probability = round(max_probability, 2)

        return final_class, max_probability * 100  # Return as percentage
