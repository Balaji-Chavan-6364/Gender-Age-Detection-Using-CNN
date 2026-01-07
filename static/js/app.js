(function () {
  // Elements
  const form = document.getElementById('detect-form');
  const modeBtns = document.querySelectorAll('.mode-btn');
  const uploadSection = document.getElementById('upload-section');

  // Camera Elements
  const cameraView = document.getElementById('camera-view');
  const imageView = document.getElementById('image-view');
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay-canvas');
  const startBtn = document.getElementById('start-camera');
  const captureBtn = document.getElementById('capture-btn');
  const liveIndicator = document.querySelector('.live-indicator');

  // Input fields
  const fileInput = document.getElementById('image');
  const hiddenCameraInput = document.getElementById('camera_image');

  // State
  let stream = null;
  let mode = 'upload';
  let liveTimer = null;
  let isProcessing = false;
  let lastDetections = [];
  let isCaptureState = false;

  // --- Mode Switching ---
  function setMode(newMode) {
    mode = newMode;
    // Update buttons
    modeBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));

    if (mode === 'upload') {
      uploadSection.classList.remove('hidden');
      cameraView.classList.add('hidden');
      imageView.classList.remove('hidden');

      stopCamera();
    } else {
      uploadSection.classList.add('hidden');
      cameraView.classList.remove('hidden');
      imageView.classList.add('hidden');

      // Auto-start camera if switching to camera mode
      // startCamera();
    }
  }

  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });


  // --- Camera Logic ---
  // Define original startCamera logic
  async function _startCameraInternal() {
    try {
      console.log('Starting camera...');
      // Reverting to simple constraints to ensure compatibility
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
      video.srcObject = stream;
      try {
        await video.play();
      } catch (e) {
        console.warn("Autoplay blocked, attempting manual play", e);
      }

      startBtn.classList.add('hidden');
      liveIndicator.classList.remove('hidden');

      // Wait for video to be ready to size canvas
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        startLiveInference();
      };

    } catch (err) {
      console.error(err);
      alert('Could not access camera. Please ensure no other app is using it and permissions are allowed. Error: ' + err.name);
    }
  }

  // Wrapper to handle Retake state
  async function startCamera() {
    if (isCaptureState) {
      // Reset UI for Retake
      video.classList.remove('hidden');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      startBtn.textContent = "Start Camera"; // Should happen naturally?
      captureBtn.classList.remove('hidden');
      isCaptureState = false;
    }
    await _startCameraInternal();
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    stopLiveInference();
    video.srcObject = null;
    startBtn.classList.remove('hidden');
    liveIndicator.classList.add('hidden');

    // Clear canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function stopLiveInference() {
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = null;
    }
    isProcessing = false;
  }

  function startLiveInference() {
    stopLiveInference();
    // Run loop every ~150ms (~6-7 FPS) to balance load
    liveTimer = setInterval(processFrame, 150);
  }

  // --- Smoothing Logic ---
  class PredictionSmoother {
    constructor(historySize = 15) { // Increase history for stability
      this.historySize = historySize;
      this.buffer = []; // [{ gender: 'M', age: '25-32' }, ...]
    }

    update(newResult) {
      this.buffer.push(newResult);
      if (this.buffer.length > this.historySize) {
        this.buffer.shift();
      }
      return this.getSmoothed();
    }

    getSmoothed() {
      if (this.buffer.length === 0) return { gender: 'Unknown', age: 'Unknown' };

      // Simple voting
      const genderCounts = {};
      const ageCounts = {};

      this.buffer.forEach(b => {
        genderCounts[b.gender] = (genderCounts[b.gender] || 0) + 1;
        ageCounts[b.age] = (ageCounts[b.age] || 0) + 1;
      });

      const getWinner = (counts) => Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);

      return {
        gender: getWinner(genderCounts),
        age: getWinner(ageCounts)
      };
    }
  }

  // Simple heuristic: if we see >0 faces, smooth the largest one. 
  const mainSmoother = new PredictionSmoother(10);


  async function processFrame() {
    if (isProcessing || !stream || video.paused || video.ended) return;

    isProcessing = true;

    try {
      // 1. Capture frame to an offscreen canvas (or reusing the overlay canvas temporarily)
      const offCanvas = document.createElement('canvas');
      offCanvas.width = video.videoWidth;
      offCanvas.height = video.videoHeight;
      const ctx = offCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      const frameData = offCanvas.toDataURL('image/jpeg', 0.5); // Lower quality for speed

      // 2. Send to backend
      const res = await fetch('/predict_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameData })
      });

      if (!res.ok) throw new Error('Server error');

      const data = await res.json();
      lastDetections = data.detections || [];

      // 3. Draw Results
      drawDetections(lastDetections);

    } catch (e) {
      console.error('Inference error:', e);
    } finally {
      isProcessing = false;
    }
  }

  function drawDetections(detections) {
    const ctx = canvas.getContext('2d');
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detections.length === 0) return;

    // Sort by size (area) to find the "main" face to smooth consistently
    detections.sort((a, b) => {
      const areaA = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
      const areaB = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
      return areaB - areaA;
    });

    // We only smooth the largest face to avoid identity swapping issues in this simple implementation
    const mainFace = detections[0];
    const smoothedParams = mainSmoother.update({ gender: mainFace.gender, age: mainFace.age });

    // Assign back to the main face for display
    mainFace.gender = smoothedParams.gender;
    mainFace.age = smoothedParams.age;

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.box;
      const width = x2 - x1;
      const height = y2 - y1;

      // Draw Box
      ctx.strokeStyle = '#00ffe5'; // Cyan
      ctx.lineWidth = 3;
      ctx.shadowColor = '#00ffe5';
      ctx.shadowBlur = 10;
      ctx.strokeRect(x1, y1, width, height);
      ctx.shadowBlur = 0;

      // Draw Label Background
      const label = `${det.gender}, ${det.age}`;
      ctx.font = 'bold 16px sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x1, y1 - 30, textWidth + 20, 30);

      // Draw Text
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1 + 10, y1 - 10);
    });
  }

  // --- Capture Snapshot ---
  async function captureSnapshot() {
    // 1. Draw the current video frame onto the visible canvas as the background
    const ctx = canvas.getContext('2d');

    // We want the image BEHIND the existing boxes (which are already on the canvas).
    ctx.globalCompositeOperation = 'destination-over';
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'source-over'; // Reset

    // 2. Save the image as a File object to the file input
    // This avoids sending huge base64 strings in the form body (fixing 413 issue)
    const offCanvas = document.createElement('canvas');
    offCanvas.width = video.videoWidth;
    offCanvas.height = video.videoHeight;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(video, 0, 0);

    offCanvas.toBlob((blob) => {
      const file = new File([blob], "camera_capture.png", { type: "image/png" });
      const container = new DataTransfer();
      container.items.add(file);
      fileInput.files = container.files;

      // Clear hidden input just in case
      hiddenCameraInput.value = '';
    }, 'image/png');

    // 3. Freeze UI
    // Stop the video and inference
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    stopLiveInference();

    // Hide video element (so only our canvas with the frozen image is visible)
    video.classList.add('hidden');
    // Ensure canvas is visible
    canvas.classList.remove('hidden');

    // Show Toast
    const toast = document.getElementById('toast-container');
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3000);

    // Update start button text for Retake
    startBtn.textContent = "Retake";
    startBtn.classList.remove('hidden');
    captureBtn.classList.add('hidden');

    isCaptureState = true;
  }


  // Event Listeners
  startBtn.addEventListener('click', () => startCamera());
  captureBtn.addEventListener('click', captureSnapshot);

  // File Preview
  fileInput.addEventListener('change', (e) => {
    if (fileInput.files && fileInput.files[0]) {
      const reader = new FileReader();
      reader.onload = (e) => {
        // We can show a preview if we want, or just rely on filename
        // For now, let's trigger a toast or something?
      }
      reader.readAsDataURL(fileInput.files[0]);
    }
  });

  // Submit Validation
  form.addEventListener('submit', (e) => {
    // In both modes, we expect fileInput to be populated now
    if (!fileInput.files[0]) {
      e.preventDefault();
      alert('Please select an image or capture a photo.');
    }
    hiddenCameraInput.value = ''; // Ensure no conflict
  });

  // Init
  setMode('upload');

})();
