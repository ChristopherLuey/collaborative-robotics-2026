
from google.cloud import speech_v1p1beta1 as speech
import rclpy
import time
from rclpy.node import Node
from tidybot_msgs.srv import AudioRecord
from tidybot_msgs.msg import TaskRequest
import numpy as np
from std_msgs.msg import String
import sounddevice as sd

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
            return ""
        
        # Step 2: Extract the transcription. Hint: we only want the first result
        for result in resp.results:
            transcript = result.alternatives[0].transcript
            break   
        #Step 3: Return the transcript
        return transcript
    

class SpeechToTextNode(Node):
    """
    to get index of sounds devices:
    python -m sounddevice list

    to activate the microphone node:
    ros2 run tidybot_control microphone_node --ros-args -p device_index:=4 -p sample_rate:=48000

    tidybot mic may be 16000 sample rate, 48000 is mine
    """

    def __init__(self):
        super().__init__('speech_to_text_node')
        self.transcriber = SpeechTranscriber(sample_rate=16000)
        self.client = self.create_client(AudioRecord, '/microphone/record')
        self.get_logger().info('Waiting for /microphone/record service...')
        if not self.client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Is the microphone node running?')
            raise RuntimeError('Service not available')
        
        self.get_logger().info('Service connected.')
        self.voice_command_pub = self.create_publisher(TaskRequest, '/task_request', 10)

        self.state = "passive" # state is either "passive" or "active"
        # passive --> means it is listening for the activation phrase
        # active --> means it is listening actively for a command.
        self.interval = {"active": 6.0, "passive": 2.0}
        self.recording = False
        self.recording_start = self.get_clock().now()

        self.audio_timer = self.create_timer(0.1, self._check_audio_cb)
        self.activation_phrase = ["hey", "robot"]

        self.rec_future = None
        self.audio_bytes = None


    def _check_audio_cb(self):
        if self.recording:
            #self.get_logger().info("recording...")
            interval = self.interval[self.state]
            now = self.get_clock().now()
            #if self.get_clock().now().nanoseconds - self.recording_start.nanoseconds > interval*10:
            if (now - self.recording_start).nanoseconds > interval*1_000_000_000:
                self.rec_future = self.stop_recording()
                if self.state == "active": self.play_tone(duration = 0.2)

            # if we are currently recording, we don't have anything to process
            return

        # if our call to stop recording is finished
        if self.rec_future is not None and self.rec_future.done():
            resp = self.rec_future.result()
            self.audio_bytes = self.process_recording(resp)
            if not self.audio_bytes:
                self.audio_bytes = None
            self.rec_future = None

        elif self.rec_future is not None and not self.rec_future.done():
            return

        self.get_logger().info("current state: " + self.state)

        # if we do not have any audio bytes, we should start recording
        if self.audio_bytes is None:

            self.recording = True
            if self.state == "active": self.play_tone(duration = 0.2)

            self.start_recording()
        
        # if we have audio bytes ready to process, and are waiting for activation phrase
        elif self.state == "passive":
            self.get_logger().info("completed recording")
            transcript = self.transcriber.transcribe_audio(self.audio_bytes)
            self.audio_bytes = None

            # if transcript is empty, just continue
            if transcript == "":
                return

            self.get_logger().info(f'Transcription: "{transcript}"') 
            stripped = transcript.strip(",.!?").lower()

            transcript_words = stripped.split(" ")

            for activation_word in self.activation_phrase:
                if activation_word not in transcript_words:
                    self.state = "passive"
                    return
                
            # we only reach this state if ALL activation words are in the transcript
            self.state = "active"

        # if we have audio bytes ready to process, and are ready to send the active query
        elif self.state == "active":
            transcript = self.transcriber.transcribe_audio(self.audio_bytes)
            self.get_logger().info("Found active query! \"" + transcript + "\"")
            self.audio_bytes = None
            self.publish_request(transcript)
            self.state = "passive"

    def start_recording(self):
        
        self.get_logger().info("requesting to start recording...")
        req = AudioRecord.Request()
        req.start = True
        self.recording_start = self.get_clock().now()
        
        future = self.client.call_async(req)

        return future
    
    def stop_recording(self):
        
        self.get_logger().info("requesting to stop recording...")
        req = AudioRecord.Request()
        req.start = False
        
        future = self.client.call_async(req)
        self.recording = False

        return future
    
    def process_recording(self, resp):
        self.get_logger().info("processing response...")
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

    def play_tone(self, frequency=440, duration=1.0, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        sd.play(wave, samplerate=sample_rate, blocking=True)
        sd.wait()  # wait until sound is finished

    def publish_request(self, request:str):
        """
        Exact task syntax is:
        
        Task 1: Pick up ____.
        Task 2: Put ___ in basket.
        Task 3: Open cabinet.

        """

        self.get_logger().info(f'Active Query: "{request}"') 
        stripped = request.strip(",.!?").lower()

        request_words = stripped.split(" ")

        obj = String()
        task = String()

        if "pick" in request_words and "up" in request_words:
            task.data = "Task1"
            up_idx = request_words.index("up")
            try:
                obj.data = request_words[up_idx + 1]
            except:
                self.get_logger().info("no object specified...")
                return

        elif "put" in request_words and "in" in request_words:
            task.data = "Task2"
            
            put_idx = request_words.index("put")
            try:
                obj.data = request_words[put_idx + 1]
            except:
                self.get_logger().info("no object specified...")
                return

        elif "open" in request_words:
            task.data = "Task3"
            obj.data = "cabinet"

        else:
            self.get_logger().info("Failed to match request to a task...") 
            return 
        
        msg = TaskRequest()
        msg.object = obj
        msg.task_type = task

        self.voice_command_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextNode()
    try:
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()