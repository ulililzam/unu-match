/**
 * Modern Modal System for UNU-Match
 * Replaces standard browser alerts and confirms with beautiful custom modals
 */

const Modal = {
    /**
     * Show a confirmation modal
     * @param {Object} config - Configuration object
     * @param {string} config.title - Modal title (optional)
     * @param {string} config.message - Modal message
     * @param {string} config.icon - Icon type: 'warning', 'info', 'success', 'error' (optional)
     * @param {string} config.confirmText - Text for confirm button (default: 'OK')
     * @param {string} config.cancelText - Text for cancel button (default: 'Batal')
     * @param {Function} config.onConfirm - Callback when confirmed
     * @param {Function} config.onCancel - Callback when cancelled (optional)
     */
    confirm: function(config) {
        return new Promise((resolve) => {
            // Create modal HTML
            const modal = document.createElement('div');
            modal.className = 'custom-modal-overlay';
            modal.innerHTML = `
                <div class="custom-modal">
                    ${config.icon ? `
                        <div class="modal-icon modal-icon-${config.icon}">
                            ${this.getIcon(config.icon)}
                        </div>
                    ` : ''}
                    ${config.title ? `<h3 class="modal-title">${config.title}</h3>` : ''}
                    <p class="modal-message">${config.message}</p>
                    <div class="modal-buttons">
                        <button class="modal-btn modal-btn-cancel">${config.cancelText || 'Batal'}</button>
                        <button class="modal-btn modal-btn-confirm">${config.confirmText || 'OK'}</button>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            // Trigger animation
            setTimeout(() => modal.classList.add('active'), 10);

            // Get buttons
            const confirmBtn = modal.querySelector('.modal-btn-confirm');
            const cancelBtn = modal.querySelector('.modal-btn-cancel');

            // Handle confirm
            confirmBtn.addEventListener('click', () => {
                this.close(modal);
                if (config.onConfirm) config.onConfirm();
                resolve(true);
            });

            // Handle cancel
            cancelBtn.addEventListener('click', () => {
                this.close(modal);
                if (config.onCancel) config.onCancel();
                resolve(false);
            });

            // Close on overlay click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.close(modal);
                    if (config.onCancel) config.onCancel();
                    resolve(false);
                }
            });

            // Close on ESC key
            const escHandler = (e) => {
                if (e.key === 'Escape') {
                    this.close(modal);
                    if (config.onCancel) config.onCancel();
                    resolve(false);
                    document.removeEventListener('keydown', escHandler);
                }
            };
            document.addEventListener('keydown', escHandler);
        });
    },

    /**
     * Show an alert modal
     * @param {Object} config - Configuration object
     * @param {string} config.title - Modal title (optional)
     * @param {string} config.message - Modal message
     * @param {string} config.icon - Icon type: 'warning', 'info', 'success', 'error' (optional)
     * @param {string} config.buttonText - Text for button (default: 'OK')
     * @param {Function} config.onClose - Callback when closed (optional)
     */
    alert: function(config) {
        return new Promise((resolve) => {
            // Create modal HTML
            const modal = document.createElement('div');
            modal.className = 'custom-modal-overlay';
            modal.innerHTML = `
                <div class="custom-modal">
                    ${config.icon ? `
                        <div class="modal-icon modal-icon-${config.icon}">
                            ${this.getIcon(config.icon)}
                        </div>
                    ` : ''}
                    ${config.title ? `<h3 class="modal-title">${config.title}</h3>` : ''}
                    <p class="modal-message">${config.message}</p>
                    <div class="modal-buttons">
                        <button class="modal-btn modal-btn-confirm single">${config.buttonText || 'OK'}</button>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            // Trigger animation
            setTimeout(() => modal.classList.add('active'), 10);

            // Get button
            const confirmBtn = modal.querySelector('.modal-btn-confirm');

            // Handle close
            const closeModal = () => {
                this.close(modal);
                if (config.onClose) config.onClose();
                resolve(true);
            };

            confirmBtn.addEventListener('click', closeModal);

            // Close on overlay click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    closeModal();
                }
            });

            // Close on ESC key
            const escHandler = (e) => {
                if (e.key === 'Escape') {
                    closeModal();
                    document.removeEventListener('keydown', escHandler);
                }
            };
            document.addEventListener('keydown', escHandler);
        });
    },

    /**
     * Close and remove modal
     */
    close: function(modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        }, 300);
    },

    /**
     * Get icon SVG based on type
     */
    getIcon: function(type) {
        const icons = {
            warning: `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>
            `,
            info: `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="12" cy="12" r="10" stroke-width="2"/>
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 16v-4m0-4h.01"/>
                </svg>
            `,
            success: `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
            `,
            error: `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="12" cy="12" r="10" stroke-width="2"/>
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 9l-6 6m0-6l6 6"/>
                </svg>
            `
        };
        return icons[type] || icons.info;
    }
};

// Export for use
window.Modal = Modal;
