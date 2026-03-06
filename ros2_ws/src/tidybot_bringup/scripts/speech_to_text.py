
from google.cloud import speech_v1p1beta1 as speech
import rclpy
import time
from rclpy.node import Node
from tidybot_msgs.srv import AudioRecord
import numpy as np
from std_msgs.msg import String

class SpeechTranscriber:
    def __init__(self, language_code='en-US', sample_rate=48000):
        """
        Initialize a SpeechTranscriber instance.

        :param language_code: The language code for transcription, e.g., 'en-US'.
        :param sample_rate: The sample rate (Hertz) of the audio file, default is 48000.
        """
        self.language_code = language_code
        self.sample_rate = sample_rate

        # Create the Speech client once
        self.client = speech.SpeechClient()

    def transcribe_audio(self, audio_content):
        """
        Uses Google Cloud Speech-to-Text to transcribe the given audio content (bytes).
        Returns the transcription as a string.

        :param audio_content: The raw bytes of the audio file to be transcribed.
        :return: A string of the combined transcription.
        """
        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        )

        # Step 1: Perform the transcription
        resp = self.client.recognize(config=config, audio=audio)

        if not resp.results:
            print("No speech detected in audio.")
            return ""
        
        # Step 2: Extract the transcription. Hint: we only want the first result
        for result in resp.results:
            transcript = result.alternatives[0].transcript
            break   
        #Step 3: Return the transcript
        return transcript
    

class SpeechToTextNode(Node):
    def __init__(self):
        super().__init__('speech_to_text_node')
        self.transcriber = SpeechTranscriber(sample_rate=48000)
        self.client = self.create_client(AudioRecord, '/microphone/record')
        self.get_logger().info('Waiting for /microphone/record service...')
        if not self.client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Is the microphone node running?')
            raise RuntimeError('Service not available')
        self.get_logger().info('Service connected.')
        self.transcription_pub = self.create_publisher(String, '/transcription', 10)
    
    def call_service(self, start: bool) -> AudioRecord.Response:
        req = AudioRecord.Request()
        req.start = start
        
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None:
            raise RuntimeError('Service call failed')
        return future.result()

    def record(self, duration: float) -> AudioRecord.Response:

        # Start
        resp = self.call_service(start=True)
        
        if not resp.success:
            self.get_logger().error(f'Start failed: {resp.message}')
            raise RuntimeError(resp.message)
        self.get_logger().info(f'Recording for {duration:.1f} seconds...')

        time.sleep(duration)

        # Stop
        resp = self.call_service(start=False)
        if not resp.success:
            self.get_logger().error(f'Stop failed: {resp.message}')
            raise RuntimeError(resp.message)
        self.get_logger().info(
            f'Got {len(resp.audio_data)} samples, '
            f'{resp.duration:.2f}s @ {resp.sample_rate} Hz'
        )
        audio_np = np.array(resp.audio_data, dtype=np.float32)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes
    
    def transcribe_and_publish(self, loop_duration:float=5.0):
        audio_bytes = self.record(duration=loop_duration)
        transcript = self.transcriber.transcribe_audio(audio_bytes)
        self.get_logger().info(f'Transcription: "{transcript}"')
        msg = String()
        msg.data = transcript
        self.transcription_pub.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextNode()
    try:
        while rclpy.ok():
            node.transcribe_and_publish(loop_duration=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()