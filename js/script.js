// ===================================
// DATA MODELS & INITIALIZATION
// ===================================

// Random Forest Model (replaces K-NN)
let rfModel = null;
let userAnswers = {};

// Program Studi Information
const prodiInfo = {
    "S1 Informatika": {
        name: "Informatika",
        fullName: "S1 Informatika",
        description: "Program studi yang mempelajari ilmu komputer, pemrograman, dan teknologi informasi. Cocok untuk kamu yang suka matematika, logika, dan ingin menjadi developer atau IT specialist.",
        subjects: ["Matematika", "Fisika", "Bahasa Inggris"],
        careers: ["Software Developer", "Data Scientist", "IT Consultant", "Network Engineer"]
    },
    "S1 Farmasi": {
        name: "Farmasi",
        fullName: "S1 Farmasi",
        description: "Mempelajari tentang obat-obatan, kesehatan, dan ilmu kimia. Cocok untuk kamu yang tertarik dengan kesehatan, suka kimia dan biologi, serta ingin membantu masyarakat dalam bidang kesehatan.",
        subjects: ["Kimia", "Biologi", "Matematika"],
        careers: ["Apoteker", "Research Scientist", "Quality Control", "Regulatory Affairs"]
    },
    "S1 Teknik Elektro": {
        name: "Teknik Elektro",
        fullName: "S1 Teknik Elektro",
        description: "Mempelajari sistem kelistrikan, elektronika, dan teknologi. Cocok untuk kamu yang suka fisika, matematika, dan ingin berkarir di bidang teknologi dan energi.",
        subjects: ["Fisika", "Matematika", "Bahasa Inggris"],
        careers: ["Electrical Engineer", "Automation Engineer", "Power System Engineer", "Electronics Designer"]
    },
    "S1 Agribisnis": {
        name: "Agribisnis",
        fullName: "S1 Agribisnis",
        description: "Mempelajari bisnis di bidang pertanian dan perkebunan. Cocok untuk kamu yang tertarik dengan bisnis, pertanian modern, dan pengembangan ekonomi pertanian.",
        subjects: ["Biologi", "Ekonomi", "Matematika"],
        careers: ["Agribusiness Manager", "Agricultural Consultant", "Agro Industry Manager", "Farm Manager"]
    },
    "S1 Akuntansi": {
        name: "Akuntansi",
        fullName: "S1 Akuntansi",
        description: "Mempelajari pencatatan keuangan, audit, dan perpajakan. Cocok untuk kamu yang teliti, suka angka, dan tertarik dengan dunia keuangan dan bisnis.",
        subjects: ["Matematika", "Ekonomi", "Bahasa Inggris"],
        careers: ["Akuntan", "Auditor", "Tax Consultant", "Financial Analyst"]
    },
    "S1 Pendidikan Bahasa Inggris": {
        name: "Pendidikan Bahasa Inggris",
        fullName: "S1 Pendidikan Bahasa Inggris",
        description: "Mempelajari bahasa Inggris dan metode pengajarannya. Cocok untuk kamu yang passionate dengan bahasa, suka mengajar, dan ingin menjadi pendidik profesional.",
        subjects: ["Bahasa Inggris", "Bahasa Indonesia", "Pendidikan"],
        careers: ["Guru Bahasa Inggris", "Translator", "Curriculum Developer", "Language Trainer"]
    },
    "S1 PGSD": {
        name: "PGSD",
        fullName: "S1 Pendidikan Guru Sekolah Dasar",
        description: "Mempelajari cara mendidik anak usia sekolah dasar. Cocok untuk kamu yang sabar, menyukai anak-anak, dan ingin menjadi guru SD yang inspiratif.",
        subjects: ["Pendidikan", "Psikologi", "Berbagai Mata Pelajaran"],
        careers: ["Guru SD", "Kepala Sekolah", "Pengembang Kurikulum", "Konsultan Pendidikan"]
    },
    "S1 Studi Islam Interdisipliner": {
        name: "Studi Islam",
        fullName: "S1 Studi Islam Interdisipliner",
        description: "Mempelajari Islam secara mendalam dan interdisipliner. Cocok untuk kamu yang ingin memahami Islam lebih dalam, suka hafalan Al-Qur'an, dan tertarik dengan kajian keislaman.",
        subjects: ["Agama Islam", "Bahasa Arab", "Sejarah"],
        careers: ["Ustadz/Ustadzah", "Peneliti Islam", "Da'i", "Konsultan Syariah"]
    },
    "S1 Manajemen": {
        name: "Manajemen",
        fullName: "S1 Manajemen",
        description: "Mempelajari cara mengelola organisasi dan bisnis. Cocok untuk kamu yang punya jiwa kepemimpinan, tertarik dengan dunia bisnis, dan ingin menjadi manajer atau entrepreneur.",
        subjects: ["Ekonomi", "Matematika", "Bahasa Inggris"],
        careers: ["Manager", "Entrepreneur", "Business Analyst", "HR Manager"]
    },
    "S1 Teknologi Hasil Pertanian": {
        name: "Teknologi Hasil Pertanian",
        fullName: "S1 Teknologi Hasil Pertanian",
        description: "Mempelajari teknologi pengolahan hasil pertanian. Cocok untuk kamu yang tertarik dengan teknologi pangan, biologi, dan inovasi di bidang pertanian.",
        subjects: ["Biologi", "Kimia", "Fisika"],
        careers: ["Food Technologist", "Quality Assurance", "Product Developer", "Food Safety Specialist"]
    }
};

// ===================================
// ML MODEL LOADING (Random Forest)
// ===================================
async function loadMLModel() {
    try {
        rfModel = new RandomForestClassifier();
        const loaded = await rfModel.loadModel('models/rf_model.json');
        
        if (loaded) {
            console.log('✅ Random Forest model loaded successfully');
            return true;
        } else {
            console.error('❌ Failed to load Random Forest model');
            return false;
        }
    } catch (error) {
        console.error('❌ Error loading ML model:', error);
        return false;
    }
}

// ===================================
// CSV PARSING & DATA LOADING (Legacy - kept for reference)
// ===================================
async function loadDataset() {
    try {
        const response = await fetch('dataset_unu.csv');
        const csvText = await response.text();
        
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        
        trainingData = lines.slice(1).map(line => {
            const values = line.split(',');
            return {
                mtk: parseFloat(values[0]),
                inggris: parseFloat(values[1]),
                agama: parseFloat(values[2]),
                fisika: parseFloat(values[3]),
                kimia: parseFloat(values[4]),
                biologi: parseFloat(values[5]),
                ekonomi: parseFloat(values[6]),
                minat_teknik: parseFloat(values[7]),
                minat_kesehatan: parseFloat(values[8]),
                minat_bisnis: parseFloat(values[9]),
                minat_pendidikan: parseFloat(values[10]),
                hafalan: parseFloat(values[11]),
                prodi: values[12].trim()
            };
        });
        
        console.log(`Dataset loaded: ${trainingData.length} records`);
        return true;
    } catch (error) {
        console.error('Error loading dataset:', error);
        return false;
    }
}

// ===================================
// STORAGE UTILITIES
// ===================================
function saveAnswers() {
    localStorage.setItem('unumatch_answers', JSON.stringify(userAnswers));
}

function loadAnswers() {
    const saved = localStorage.getItem('unumatch_answers');
    return saved ? JSON.parse(saved) : null;
}

function clearAnswers() {
    localStorage.removeItem('unumatch_answers');
    localStorage.removeItem('unumatch_results');
}

function saveResults(results) {
    localStorage.setItem('unumatch_results', JSON.stringify(results));
}

function loadResults() {
    const saved = localStorage.getItem('unumatch_results');
    return saved ? JSON.parse(saved) : null;
}

// ===================================
// CLUSTERING ALGORITHM (K-NN Based)
// ===================================
function normalizeValue(value, min, max) {
    return (value - min) / (max - min);
}

function calculateEuclideanDistance(point1, point2) {
    const features = [
        'mtk', 'inggris', 'agama', 'fisika', 'kimia', 
        'biologi', 'ekonomi', 'minat_teknik', 'minat_kesehatan', 
        'minat_bisnis', 'minat_pendidikan', 'hafalan'
    ];
    
    let sum = 0;
    
    // Normalize and calculate distance
    features.forEach(feature => {
        let val1 = point1[feature];
        let val2 = point2[feature];
        
        // Normalize based on feature type
        if (['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi'].includes(feature)) {
            // Nilai mata pelajaran: 0-100
            val1 = val1 / 100;
            val2 = val2 / 100;
        } else if (['minat_teknik', 'minat_kesehatan', 'minat_bisnis', 'minat_pendidikan'].includes(feature)) {
            // Minat: 1-5
            val1 = (val1 - 1) / 4;
            val2 = (val2 - 1) / 4;
        }
        // hafalan sudah 0-1, tidak perlu normalisasi
        
        const diff = val1 - val2;
        sum += diff * diff;
    });
    
    return Math.sqrt(sum);
}

function findTopMatches(userProfile, k = 100) {
    // Calculate distance to all training data
    const distances = trainingData.map(data => ({
        prodi: data.prodi,
        distance: calculateEuclideanDistance(userProfile, data)
    }));
    
    // Sort by distance (closest first)
    distances.sort((a, b) => a.distance - b.distance);
    
    // Get top k nearest neighbors
    const topK = distances.slice(0, k);
    
    // Count votes for each prodi
    const votes = {};
    topK.forEach(item => {
        if (!votes[item.prodi]) {
            votes[item.prodi] = { count: 0, totalDistance: 0 };
        }
        votes[item.prodi].count++;
        votes[item.prodi].totalDistance += item.distance;
    });
    
    // Calculate match percentage
    const results = Object.keys(votes).map(prodi => {
        const avgDistance = votes[prodi].totalDistance / votes[prodi].count;
        const matchPercentage = Math.max(0, 100 - (avgDistance * 50)); // Scale to 0-100%
        
        return {
            prodi: prodi,
            matchPercentage: Math.round(matchPercentage * 10) / 10,
            votes: votes[prodi].count,
            avgDistance: avgDistance
        };
    });
    
    // Sort by match percentage
    results.sort((a, b) => b.matchPercentage - a.matchPercentage);
    
    return results;
}

// ===================================
// PREDICTION ALGORITHM (Random Forest)
// ===================================
function predictProdi() {
    if (!userAnswers || Object.keys(userAnswers).length === 0) {
        console.error('No user answers found');
        return null;
    }
    
    if (!rfModel || !rfModel.isLoaded) {
        console.error('Random Forest model not loaded');
        return null;
    }
    
    // Build user profile from answers
    const userProfile = {
        mtk: userAnswers.mtk || 50,
        inggris: userAnswers.inggris || 50,
        agama: userAnswers.agama || 50,
        fisika: userAnswers.fisika || 50,
        kimia: userAnswers.kimia || 50,
        biologi: userAnswers.biologi || 50,
        ekonomi: userAnswers.ekonomi || 50,
        minat_teknik: userAnswers.minat_teknik || 3,
        minat_kesehatan: userAnswers.minat_kesehatan || 3,
        minat_bisnis: userAnswers.minat_bisnis || 3,
        minat_pendidikan: userAnswers.minat_pendidikan || 3,
        hafalan: userAnswers.hafalan || 0
    };
    
    console.log('User profile:', userProfile);
    
    // Get prediction from Random Forest
    const prediction = rfModel.predict(userProfile);
    console.log('Random Forest prediction:', prediction);
    
    // Format results
    const results = prediction.topPredictions.map((pred, index) => ({
        rank: pred.rank,
        prodi: pred.prodi,
        matchPercentage: pred.matchPercentage,
        confidence: pred.confidence,
        featureContributions: index === 0 ? prediction.featureContributions : null,
        info: prodiInfo[pred.prodi] || {
            name: pred.prodi,
            fullName: pred.prodi,
            description: "Program studi yang sesuai dengan profil Anda.",
            subjects: [],
            careers: []
        }
    }));
    
    return results;
}

// ===================================
// UI UTILITIES
// ===================================
function updateSliderValue(sliderId, value) {
    const valueDisplay = document.getElementById(`${sliderId}-value`);
    if (valueDisplay) {
        valueDisplay.textContent = value;
    }
    userAnswers[sliderId] = parseInt(value);
    saveAnswers();
}

function setRating(questionId, rating) {
    userAnswers[questionId] = rating;
    saveAnswers();
    
    // Update UI
    const stars = document.querySelectorAll(`#${questionId} .star`);
    stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('filled');
        } else {
            star.classList.remove('filled');
        }
    });
}

function toggleHafalan(value) {
    userAnswers.hafalan = value ? 1 : 0;
    saveAnswers();
}

function updateProgressBar(current, total) {
    const percentage = (current / total) * 100;
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        progressBar.style.width = percentage + '%';
    }
    
    const progressText = document.getElementById('progress-text');
    if (progressText) {
        progressText.textContent = `Langkah ${current} dari ${total}`;
    }
}

function showStep(stepNumber) {
    const steps = document.querySelectorAll('.question-step');
    steps.forEach((step, index) => {
        if (index + 1 === stepNumber) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
    
    updateProgressBar(stepNumber, steps.length);
    
    // Update button visibility
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const submitBtn = document.getElementById('submit-btn');
    
    if (prevBtn) prevBtn.style.display = stepNumber === 1 ? 'none' : 'block';
    if (nextBtn) nextBtn.style.display = stepNumber === steps.length ? 'none' : 'block';
    if (submitBtn) submitBtn.style.display = stepNumber === steps.length ? 'block' : 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function validateCurrentStep(stepNumber) {
    // Basic validation - ensure at least some input
    switch(stepNumber) {
        case 1: // Nilai mata pelajaran
            return userAnswers.mtk !== undefined &&
                   userAnswers.inggris !== undefined &&
                   userAnswers.agama !== undefined &&
                   userAnswers.fisika !== undefined &&
                   userAnswers.kimia !== undefined &&
                   userAnswers.biologi !== undefined &&
                   userAnswers.ekonomi !== undefined;
        
        case 2: // Minat
            return userAnswers.minat_teknik !== undefined &&
                   userAnswers.minat_kesehatan !== undefined &&
                   userAnswers.minat_bisnis !== undefined &&
                   userAnswers.minat_pendidikan !== undefined;
        
        case 3: // Hafalan
            return userAnswers.hafalan !== undefined;
        
        default:
            return true;
    }
}

function showNotification(message, type = 'info') {
    // Simple alert for now
    alert(message);
}

// ===================================
// RESULT DISPLAY
// ===================================
function displayResults(results) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = '';
    
    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'result-card fade-in';
        
        const rankEmoji = result.rank === 1 ? '🥇' : result.rank === 2 ? '🥈' : '🥉';
        
        // Feature contributions for top result
        let featureExplanation = '';
        if (result.featureContributions) {
            featureExplanation = `
                <div style="margin-top: 1rem; padding: 1rem; background: #f0fdf4; border-radius: 0.5rem;">
                    <strong style="color: #16a34a;">🔍 Mengapa jurusan ini cocok?</strong>
                    <div style="margin-top: 0.5rem; font-size: 0.875rem;">
                        ${result.featureContributions.slice(0, 3).map(fc => {
                            const label = rfModel.getFeatureLabel(fc.feature);
                            return `<div style="margin: 0.25rem 0;">• <strong>${label}</strong>: ${fc.value} (kontribusi ${(fc.contribution * 100).toFixed(1)}%)</div>`;
                        }).join('')}
                    </div>
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="result-rank">${rankEmoji}</div>
            <h2 class="result-title">${result.info.fullName}</h2>
            <div class="result-match">Kecocokan: ${result.matchPercentage}%</div>
            <div class="match-bar">
                <div class="match-fill" style="width: ${result.matchPercentage}%"></div>
            </div>
            ${result.confidence ? `<div style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">Tingkat Keyakinan: <strong>${result.confidence}</strong></div>` : ''}
            <p class="result-description">${result.info.description}</p>
            ${featureExplanation}
            
            ${result.info.subjects && result.info.subjects.length > 0 ? `
                <div style="margin-top: 1rem;">
                    <strong>Mata Pelajaran Utama:</strong>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                        ${result.info.subjects.map(s => `<span class="badge badge-green">${s}</span>`).join('')}
                    </div>
                </div>
            ` : ''}
            
            ${result.info.careers && result.info.careers.length > 0 ? `
                <div style="margin-top: 1rem;">
                    <strong>Prospek Karir:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem; color: var(--text-gray);">
                        ${result.info.careers.map(c => `<li>${c}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
        
        resultsContainer.appendChild(card);
    });
}

// ===================================
// INITIALIZATION
// ===================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('UNU-Match initialized');
    
    // Load Random Forest model
    const modelLoaded = await loadMLModel();
    if (!modelLoaded) {
        console.error('Failed to load ML model');
        // Fallback: could show error message or use legacy K-NN
    }
    
    // Load saved answers if any
    const savedAnswers = loadAnswers();
    if (savedAnswers) {
        userAnswers = savedAnswers;
        console.log('Loaded saved answers:', userAnswers);
    }
});

// ===================================
// EXPORT FOR BROWSER
// ===================================
if (typeof window !== 'undefined') {
    window.unumatch = {
        loadMLModel,
        loadDataset,  // Keep for legacy
        saveAnswers,
        loadAnswers,
        clearAnswers,
        saveResults,
        loadResults,
        predictProdi,
        updateSliderValue,
        setRating,
        toggleHafalan,
        showStep,
        validateCurrentStep,
        showNotification,
        displayResults,
        prodiInfo,
        userAnswers,
        rfModel
    };
}
