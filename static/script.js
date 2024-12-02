/** static/script.js **/

const LONG_PRESS_DURATION = 500;
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
        this.containerId = containerId; // Store the container ID for unique overlay
        this.bindEventListeners();
    }
  
    bindEventListeners() {
        // Create an invisible overlay div with unique ID for detecting interactions when hidden
        this.overlay = document.createElement('div');
        this.overlay.id = `overlay-${this.containerId}`; // Add unique ID to overlay
        this.overlay.style.cssText = `
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 300px;
            height: 400px;
            display: none;
            z-index: 3;
            ${this.containerId === 'text-block-container' ? 'left: 0%;' : 'right: 0%;'}
        `;
    
        document.body.appendChild(this.overlay);
  
        const handleEvent = (eventName, handler) => {
            const boundHandler = (e) => {
                const target = e.target;
                const isThisContainer = target.closest(`#${this.containerId}`);
                const isThisOverlay = target.closest(`#overlay-${this.containerId}`);
                
                if (isThisContainer || isThisOverlay) {
                    e.stopPropagation();
                    handler.call(this, e);
                }
            };

            this.container.addEventListener(eventName, boundHandler);
            this.overlay.addEventListener(eventName, boundHandler);
        };

        handleEvent('touchstart', this.handleTouchStart);
        handleEvent('touchend', this.handleTouchEnd);
        handleEvent('mousedown', this.handleMouseDown);
        handleEvent('mouseup', this.handleMouseUp);
    }

    toggleVisibility() {
        this.isVisible = !this.isVisible;
    
        if (this.isVisible) {
            this.container.classList.remove('hidden');
            this.overlay.style.display = 'none';
            this.container.style.zIndex = '2';
            this.overlay.classList.remove('touch-indicator');
        } else {
            this.container.classList.add('hidden');
            this.overlay.style.display = 'block';
            this.container.style.zIndex = '1';
            this.overlay.classList.add('touch-indicator');
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

    showTextBlock(index) {
        this.blocks.forEach((block, i) => {
            block.classList.toggle('active', i === index);
            block.classList.toggle('inactive', i !== index);
        });
        this.currentIndex = index;
    }
  }

class DetectionHistory extends TextBlock {
  constructor(containerId, blocks, checklistId) {
      super(containerId, blocks);
      this.checklist = document.getElementById(checklistId);
      /*this.allItems = Array.from({ length: 9 }).map((_, i) => this.createChecklistItem(`Item ${i + 1}`));*/
      this.allItems = Array.from(this.checklist.querySelectorAll('li'));
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
        this.currentObject = null;
        
        this.leftTextBlock = new TextBlock('text-block-container', 
            document.querySelectorAll('#text-block-container .text-block'));
        this.detectionHistory = new DetectionHistory('right-text-block-container',
            document.querySelectorAll('#right-text-block-container .text-block'),
            'checklist');
        
        // Content data for different objects
        this.contentData = {
            yogurt: {
                title: "Greek Yogurt",
                info1: "Name: Greek Yogurt<br>Brand: Kirkland<br>Flavor: Greek<br>Price: $13.69",
                info2: "Here goes additional yogurt information",
                info3: "Relevant yogurt nutritional information"
            },
            curry: {
                title: "Curry Powder",
                info1: "Name: Curry Powder<br>Brand: McCormick<br>Spice Level: Medium<br>Price: $7.49",
                info2: "Here goes additional curry information",
                info3: "Relevant curry nutritional information"
            },
            honey: {
                title: "Honey",
                info1: "Name: Organic Honey<br>Brand: Nature Nate's<br>Type: Raw<br>Price: $8.99",
                info2: "Here goes additional honey information",
                info3: "Relevant honey nutritional information"
            }
        };
        
        // Add default content
        this.contentData.default = {
            title: "Looking for objects...",
            info1: "Point your camera at some items to get started",
            info2: "Point your camera at some items to get started", 
            info3: "Point your camera at some items to get started"
        };
        
        this.setupEventSource();
        this.setupIconListeners();
    }

    setupIconListeners() {
        const iconYogurt = document.getElementById('icon-yogurt');
        const iconCurry = document.getElementById('icon-curry');
        const iconHoney = document.getElementById('icon-honey');

        iconYogurt.addEventListener('click', () => this.updateContent('yogurt'));
        iconCurry.addEventListener('click', () => this.updateContent('curry'));
        iconHoney.addEventListener('click', () => this.updateContent('honey'));
    }

    updateContent(objectType) {
        const content = objectType === 'None' ? this.contentData.default : this.contentData[objectType];
        if (!content) return;

        const textBlocks = document.querySelectorAll('.text-block');
        textBlocks[0].querySelector('h1').innerText = content.title;
        textBlocks[0].querySelector('p').innerHTML = content.info1;
        textBlocks[1].querySelector('h1').innerText = content.title;
        textBlocks[1].querySelector('p').innerHTML = content.info2;
        textBlocks[2].querySelector('h1').innerText = content.title;
        textBlocks[2].querySelector('p').innerHTML = content.info3;

        this.setActiveIcon(objectType);
    }

    setActiveIcon(activeType) {
        const icons = document.querySelectorAll('#icon-nav .icon');
        icons.forEach(icon => {
            icon.classList.toggle('active', icon.id === `icon-${activeType}`);
        });
    }

    setupEventSource() {
        const eventSource = new EventSource('/video_feed');

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update video feed
            this.videoFeed.src = `data:image/png;base64,${data.frame}`;
            
            // Handle closest object detection
            if (data.closest_object) {
                const objectName = data.closest_object.name;
                if (objectName !== this.currentObject) {
                    this.currentObject = objectName;
                    this.updateContent(objectName === 'None' ? 'default' : objectName);
                }
            } else {
                // No closest object, show default
                this.currentObject = null;
                this.updateContent('default');
            }
            
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

document.addEventListener('DOMContentLoaded', () => {
    const iconYogurt = document.getElementById('icon-yogurt');
    const iconCurry = document.getElementById('icon-curry');
    const iconHoney = document.getElementById('icon-honey');

    const textBlocks = document.querySelectorAll('.text-block');

    // Define content for each item (yogurt, curry, honey)
    const contentData = {
        yogurt: {
            title: "Greek Yogurt",
            info1: "Name: Greek Yogurt<br>Brand: Kirkland<br>Flavor: Greek<br>Price: $13.69",
            info2: "Here goes additional yogurt information",
            info3: "Relevant yogurt nutritional information"
        },
        curry: {
            title: "Curry Powder",
            info1: "Name: Curry Powder<br>Brand: McCormick<br>Spice Level: Medium<br>Price: $7.49",
            info2: "Here goes additional curry information",
            info3: "Relevant curry nutritional information"
        },
        honey: {
            title: "Honey",
            info1: "Name: Organic Honey<br>Brand: Nature Nate's<br>Type: Raw<br>Price: $8.99",
            info2: "Here goes additional honey information",
            info3: "Relevant honey nutritional information"
        }
    };

    // Function to update all text blocks with selected content
    function updateTextBlocks(content) {
        // Update each block's content with the relevant information
        textBlocks[0].querySelector('h1').innerText = content.title;
        textBlocks[0].querySelector('p').innerHTML = content.info1;
        textBlocks[1].querySelector('h1').innerText = content.title;
        textBlocks[1].querySelector('p').innerHTML = content.info2;
        textBlocks[2].querySelector('h1').innerText = content.title;
        textBlocks[2].querySelector('p').innerHTML = content.info3;
    }

    // Event listeners to switch content when icons are clicked
    iconYogurt.addEventListener('click', () => {
        updateTextBlocks(contentData.yogurt);
        setActiveIcon('yogurt');
    });
    iconCurry.addEventListener('click', () => {
        updateTextBlocks(contentData.curry);
        setActiveIcon('curry');
    });
    iconHoney.addEventListener('click', () => {
        updateTextBlocks(contentData.honey);
        setActiveIcon('honey');
    });

    // Function to set the active icon visually
    function setActiveIcon(activeType) {
        [iconYogurt, iconCurry, iconHoney].forEach(icon => {
            icon.classList.toggle('active', icon.id === `icon-${activeType}`);
        });
    }
    // Tutorial handling
    const tutorialOverlay = document.getElementById('tutorial-overlay');
    const tutorialDismiss = document.getElementById('tutorial-dismiss');

    // Check if tutorial has been shown before
    const hasSeenTutorial = localStorage.getItem('hasSeenTutorial');

    if (!hasSeenTutorial) {
        tutorialOverlay.classList.remove('hidden');
    }

    tutorialDismiss.addEventListener('click', () => {
        tutorialOverlay.classList.add('hidden');
        localStorage.setItem('hasSeenTutorial', 'true');
    });
});