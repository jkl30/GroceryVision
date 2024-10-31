/** static/script.js **/

const LONG_PRESS_DURATION = 1500;
const SWIPE_THRESHOLD = 50;
const MAX_HISTORY_ITEMS = 9; // 9 items total for 3 pages of 3 items each
const ITEMS_PER_PAGE = 3;

/**
 * TextBlock Component
 * Represents a reusable text block component
 */
class TextBlock {
  constructor(containerId, blocks) {
      this.container = document.getElementById(containerId);
      this.blocks = blocks;
      this.currentIndex = 0;
      this.longPressTimeout = null;
      this.startY = 0;
      this.startTime = 0;
      this.isVisible = true;
      this.bindEventListeners();
  }

  bindEventListeners() {
      // Create an invisible overlay div for detecting interactions when hidden
      this.overlay = document.createElement('div');
      this.overlay.style.cssText = `
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          display: none;
          z-index: 1;
      `;
      this.container.parentNode.insertBefore(this.overlay, this.container);

      // Bind events to both container and overlay
      [this.container, this.overlay].forEach(element => {
          element.addEventListener('touchstart', this.handleTouchStart.bind(this));
          element.addEventListener('touchend', this.handleTouchEnd.bind(this));
          element.addEventListener('mousedown', this.handleMouseDown.bind(this));
          element.addEventListener('mouseup', this.handleMouseUp.bind(this));
      });
  }

  showTextBlock(index) {
      this.blocks.forEach((block, i) => {
          block.classList.toggle('active', i === index);
          block.classList.toggle('inactive', i !== index);
      });
      this.currentIndex = index;
  }

  toggleVisibility() {
      this.isVisible = !this.isVisible;
      this.container.classList.toggle('hidden', !this.isVisible);
      this.overlay.style.display = this.isVisible ? 'none' : 'block';
  }

  handleSwipe(deltaY) {
      if (Math.abs(deltaY) > SWIPE_THRESHOLD) {
          if (deltaY > 0) {
              this.currentIndex = (this.currentIndex + 1) % this.blocks.length;
          } else {
              this.currentIndex = (this.currentIndex - 1 + this.blocks.length) % this.blocks.length;
          }
          this.showTextBlock(this.currentIndex);
      }
  }

  handleTouchStart(event) {
      this.startY = event.touches[0].clientY;
      this.startTime = Date.now();
      this.longPressTimeout = setTimeout(() => {
          this.toggleVisibility();
      }, LONG_PRESS_DURATION);
  }

  handleTouchEnd(event) {
      clearTimeout(this.longPressTimeout);
      if (Date.now() - this.startTime < LONG_PRESS_DURATION) {
          const deltaY = event.changedTouches[0].clientY - this.startY;
          this.handleSwipe(deltaY);
      }
  }

  handleMouseDown(event) {
      this.startY = event.clientY;
      this.startTime = Date.now();
      this.longPressTimeout = setTimeout(() => {
          this.toggleVisibility();
      }, LONG_PRESS_DURATION);
  }

  handleMouseUp(event) {
      clearTimeout(this.longPressTimeout);
      if (Date.now() - this.startTime < LONG_PRESS_DURATION) {
          const deltaY = event.clientY - this.startY;
          this.handleSwipe(deltaY);
      }
  }
}

class DetectionHistory extends TextBlock {
  constructor(containerId, blocks, checklistId) {
      super(containerId, blocks);
      this.checklist = document.getElementById(checklistId);
      this.allItems = Array.from({ length: 9 }).map((_, i) => this.createChecklistItem(`Item ${i + 1}`));
      this.itemPages = this.createPages();
      this.currentPage = 0;
      this.updateView();
  }

  createChecklistItem(text) {
      const item = document.createElement('li');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.id = `item-${Math.random().toString(36).substr(2, 9)}`;

      const label = document.createElement('label');
      label.htmlFor = checkbox.id;
      label.textContent = text;

      item.appendChild(checkbox);
      item.appendChild(label);
      return item;
  }

  createPages() {
      const pages = [];
      for (let i = 0; i < this.allItems.length; i += ITEMS_PER_PAGE) {
          pages.push(this.allItems.slice(i, i + ITEMS_PER_PAGE));
      }
      return pages;
  }

  updateView() {
      this.checklist.innerHTML = '';
      const currentItems = this.itemPages[this.currentPage];
      currentItems.forEach(item => {
          this.checklist.appendChild(item.cloneNode(true));
      });

      // Update dots
      const dots = this.blocks[0].querySelectorAll('.dot');
      dots.forEach((dot, index) => {
          dot.classList.toggle('active', index === this.currentPage);
      });
  }

  handleSwipe(deltaY) {
      if (Math.abs(deltaY) > SWIPE_THRESHOLD) {
          if (deltaY > 0) {
              this.currentPage = (this.currentPage + 1) % this.itemPages.length;
          } else {
              this.currentPage = (this.currentPage - 1 + this.itemPages.length) % this.itemPages.length;
          }
          this.updateView();
      }
  }

  addDetection(detection) {
      const timestamp = new Date().toLocaleTimeString();
      const newItem = this.createChecklistItem(
          `${detection.class_name} (${detection.confidence.toFixed(2)}) - ${timestamp}`
      );

      // Remove oldest item if we're at capacity
      if (this.allItems.length >= MAX_HISTORY_ITEMS) {
          this.allItems.pop();
      }

      // Add new item to the beginning
      this.allItems.unshift(newItem);

      // Recreate pages and update view
      this.itemPages = this.createPages();
      this.updateView();
  }
}

/**
 * DetectionInterface Class
 * Manages the detection interface and updates
 */
class DetectionInterface {
    constructor() {
        this.videoFeed = document.getElementById('video-feed');
        this.detectionsDiv = document.getElementById('detections');
        this.statsDiv = document.getElementById('stats');
        this.ocrTextDiv = document.getElementById('ocr-text');
        
        this.leftTextBlock = new TextBlock('text-block-container', 
            document.querySelectorAll('#text-block-container .text-block'));
        this.detectionHistory = new DetectionHistory('right-text-block-container',
            document.querySelectorAll('#right-text-block-container .text-block'),
            'checklist');
        
        this.setupEventSource();
    }

    setupEventSource() {
        const eventSource = new EventSource('/video_feed');

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update video feed
            this.videoFeed.src = `data:image/png;base64,${data.frame}`;
            
            // Update detections
            this.updateDetections(data.detections);
            
            // Update OCR text
            if (data.ocr_text) {
                this.ocrTextDiv.innerHTML = `<p>${data.ocr_text}</p>`;
            }
            
            // Update stats
            this.updateStats(data);
        };

        eventSource.onerror = (error) => {
            console.error('EventSource failed:', error);
            eventSource.close();
        };
    }

    updateDetections(detections) {
        this.detectionsDiv.innerHTML = `
            <ul>
                ${detections.map(det => `
                    <li>${det.class_name}: ${det.confidence.toFixed(2)}</li>
                `).join('')}
            </ul>
        `;
        
        // Add to history
        detections.forEach(det => {
            this.detectionHistory.addDetection(det);
        });
    }

    updateStats(data) {
        const stats = {
            totalDetections: data.detections.length,
            averageConfidence: data.detections.reduce((acc, det) => acc + det.confidence, 0) / 
                (data.detections.length || 1),
            timestamp: new Date().toLocaleString()
        };
        
        this.statsDiv.innerHTML = `
            <p>Total Detections: ${stats.totalDetections}</p>
            <p>Average Confidence: ${stats.averageConfidence.toFixed(2)}%</p>
            <p>Last Update: ${stats.timestamp}</p>
        `;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new DetectionInterface();
});