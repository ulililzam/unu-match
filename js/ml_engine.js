// ===================================
// UNU-Match ML Engine
// Random Forest Inference Engine for JavaScript
// ===================================

/**
 * Random Forest Classifier for JavaScript
 * Loads pre-trained model from Python and performs predictions
 */
class RandomForestClassifier {
    constructor() {
        this.model = null;
        this.featureNames = null;
        this.classNames = null;
        this.featureImportance = null;
        this.metadata = null;
        this.isLoaded = false;
    }

    /**
     * Load Random Forest model from JSON
     */
    async loadModel(modelPath = '../models/rf_model.json') {
        try {
            console.log('🌲 Loading Random Forest model...');
            
            // Load model structure
            const modelResponse = await fetch(modelPath);
            this.model = await modelResponse.json();
            
            // Load feature importance
            const importanceResponse = await fetch('../models/feature_importance.json');
            this.featureImportance = await importanceResponse.json();
            
            // Load metadata
            const metadataResponse = await fetch('../models/model_metadata.json');
            this.metadata = await metadataResponse.json();
            
            this.featureNames = this.model.feature_names;
            this.classNames = this.model.class_names;
            
            console.log(`✅ Model loaded successfully!`);
            console.log(`   - Model Type: ${this.model.model_type}`);
            console.log(`   - Trees: ${this.model.n_estimators}`);
            console.log(`   - Features: ${this.featureNames.length}`);
            console.log(`   - Classes: ${this.classNames.length}`);
            console.log(`   - Test Accuracy: ${(this.metadata.metrics.test_accuracy * 100).toFixed(2)}%`);
            
            this.isLoaded = true;
            return true;
        } catch (error) {
            console.error('❌ Error loading model:', error);
            return false;
        }
    }

    /**
     * Predict class for a single sample
     * @param {Object} features - Feature values (e.g., {mtk: 90, inggris: 85, ...})
     * @returns {Object} Prediction results with probabilities
     */
    predict(features) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Convert features object to array in correct order
        const featureArray = this.featureNames.map(name => features[name]);

        // Normalize features
        const normalizedFeatures = this.normalizeFeatures(featureArray);

        // Get predictions from all trees
        const treePredictions = this.model.trees.map(tree => 
            this.predictTree(tree, normalizedFeatures)
        );

        // Count votes for each class with weighted voting
        const votes = new Array(this.classNames.length).fill(0);
        const weightedVotes = new Array(this.classNames.length).fill(0);
        
        treePredictions.forEach(prediction => {
            votes[prediction]++;
            // Weight vote by feature strength
            const weight = this.calculatePredictionWeight(features, this.classNames[prediction]);
            weightedVotes[prediction] += weight;
        });

        // Calculate probabilities (use weighted votes for better accuracy)
        const totalWeight = weightedVotes.reduce((a, b) => a + b, 0);
        const probabilities = weightedVotes.map(w => w / totalWeight);

        // Get top 3 predictions
        const sortedIndices = probabilities
            .map((prob, idx) => ({prob, idx}))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 3);

        const topPredictions = sortedIndices.map((item, rank) => ({
            rank: rank + 1,
            prodi: this.classNames[item.idx],
            probability: item.prob,
            matchPercentage: Math.round(item.prob * 100 * 10) / 10,
            confidence: this.calculateConfidence(item.prob, probabilities)
        }));

        return {
            topPredictions,
            allProbabilities: probabilities,
            featureContributions: this.explainPrediction(features, topPredictions[0].prodi)
        };
    }

    /**
     * Predict using a single decision tree
     * @param {Object} tree - Tree structure
     * @param {Array} features - Normalized feature array
     * @returns {number} Class index
     */
    predictTree(tree, features) {
        let node = tree;

        while (node.type === 'decision') {
            const featureValue = features[node.feature];
            if (featureValue <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }

        return node.prediction;
    }

    /**
     * Normalize features to 0-1 range
     * @param {Array} features - Raw feature values
     * @returns {Array} Normalized features
     */
    normalizeFeatures(features) {
        return features.map((value, idx) => {
            const featureName = this.featureNames[idx];
            
            // Normalize based on feature type
            if (['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi'].includes(featureName)) {
                // Grades: 0-100 → 0-1
                return value / 100;
            } else if (['minat_teknik', 'minat_kesehatan', 'minat_bisnis', 'minat_pendidikan'].includes(featureName)) {
                // Interest: 1-5 → 0-1
                return (value - 1) / 4;
            } else {
                // Hafalan: already 0-1
                return value;
            }
        });
    }

    /**
     * Calculate prediction weight based on feature alignment
     * @param {Object} features - User features
     * @param {string} prodi - Program studi name
     * @returns {number} Weight multiplier (0.5 - 1.5)
     */
    calculatePredictionWeight(features, prodi) {
        let weight = 1.0;
        
        // Boost weight based on strong feature alignment
        if (prodi.includes('Informatika')) {
            if (features.minat_teknik >= 4) weight += 0.3;
            if (features.mtk >= 80) weight += 0.2;
        } else if (prodi.includes('Farmasi')) {
            if (features.minat_kesehatan >= 4) weight += 0.3;
            if (features.kimia >= 80 && features.biologi >= 80) weight += 0.2;
        } else if (prodi.includes('Teknik Elektro')) {
            if (features.minat_teknik >= 4) weight += 0.3;
            if (features.fisika >= 80) weight += 0.2;
        } else if (prodi.includes('Studi Islam')) {
            if (features.agama >= 85) weight += 0.3;
            if (features.hafalan === 1) weight += 0.2;
        } else if (prodi.includes('Akuntansi') || prodi.includes('Manajemen')) {
            if (features.minat_bisnis >= 4) weight += 0.3;
            if (features.ekonomi >= 80) weight += 0.2;
        } else if (prodi.includes('PGSD') || prodi.includes('Pendidikan')) {
            if (features.minat_pendidikan >= 4) weight += 0.3;
            if (prodi.includes('Inggris') && features.inggris >= 85) weight += 0.2;
        } else if (prodi.includes('Agribisnis')) {
            if (features.minat_bisnis >= 3 && features.biologi >= 75) weight += 0.3;
        } else if (prodi.includes('Teknologi Hasil Pertanian')) {
            if (features.biologi >= 80 && features.kimia >= 75) weight += 0.3;
        }
        
        // Penalize if critical features are too low
        if (prodi.includes('Informatika') && features.mtk < 60) weight *= 0.7;
        if (prodi.includes('Farmasi') && (features.kimia < 60 || features.biologi < 60)) weight *= 0.7;
        if (prodi.includes('Studi Islam') && features.agama < 70) weight *= 0.6;
        
        return Math.max(0.5, Math.min(1.5, weight)); // Clamp between 0.5-1.5
    }

    /**
     * Calculate confidence score for prediction
     * @param {number} topProbability - Probability of top prediction
     * @param {Array} allProbabilities - All class probabilities
     * @returns {string} Confidence level
     */
    calculateConfidence(topProbability, allProbabilities) {
        const sortedProbs = [...allProbabilities].sort((a, b) => b - a);
        const gap = sortedProbs[0] - sortedProbs[1];

        // Improved confidence thresholds
        if (topProbability >= 0.55 && gap >= 0.25) return 'Sangat Tinggi';
        if (topProbability >= 0.40 && gap >= 0.18) return 'Tinggi';
        if (topProbability >= 0.28 && gap >= 0.10) return 'Sedang';
        return 'Rendah';
    }

    /**
     * Explain prediction by showing feature contributions
     * @param {Object} features - User input features
     * @param {string} predictedClass - Predicted class name
     * @returns {Array} Feature contributions
     */
    explainPrediction(features, predictedClass) {
        // Get feature importance for explanation
        const contributions = this.featureImportance.features.map(item => {
            const featureName = item.feature;
            const importance = item.importance;
            const userValue = features[featureName];
            
            // Simple contribution score (importance * normalized value)
            let normalizedValue;
            if (['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi'].includes(featureName)) {
                normalizedValue = userValue / 100;
            } else if (['minat_teknik', 'minat_kesehatan', 'minat_bisnis', 'minat_pendidikan'].includes(featureName)) {
                normalizedValue = (userValue - 1) / 4;
            } else {
                normalizedValue = userValue;
            }
            
            const contribution = importance * normalizedValue;
            
            return {
                feature: featureName,
                value: userValue,
                importance: importance,
                contribution: contribution
            };
        });

        // Sort by contribution
        contributions.sort((a, b) => b.contribution - a.contribution);

        return contributions.slice(0, 5); // Top 5 features
    }

    /**
     * Get human-readable feature name
     * @param {string} featureName - Feature code
     * @returns {string} Human-readable name
     */
    getFeatureLabel(featureName) {
        const labels = {
            'mtk': 'Matematika',
            'inggris': 'Bahasa Inggris',
            'agama': 'Pendidikan Agama',
            'fisika': 'Fisika',
            'kimia': 'Kimia',
            'biologi': 'Biologi',
            'ekonomi': 'Ekonomi',
            'minat_teknik': 'Minat Teknik',
            'minat_kesehatan': 'Minat Kesehatan',
            'minat_bisnis': 'Minat Bisnis',
            'minat_pendidikan': 'Minat Pendidikan',
            'hafalan': 'Kemampuan Hafalan'
        };
        return labels[featureName] || featureName;
    }

    /**
     * Get model metadata
     * @returns {Object} Model metadata
     */
    getMetadata() {
        return this.metadata;
    }

    /**
     * Get feature importance
     * @returns {Array} Feature importance list
     */
    getFeatureImportance() {
        return this.featureImportance.features;
    }
}

// ===================================
// EXPORT
// ===================================
if (typeof window !== 'undefined') {
    window.RandomForestClassifier = RandomForestClassifier;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RandomForestClassifier;
}
