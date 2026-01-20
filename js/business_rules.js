// ===================================
// UNU-MATCH BUSINESS RULES ENGINE
// Validates ML predictions with logical constraints
// ===================================

/**
 * Business Rules Engine
 * Applies domain knowledge and prerequisite constraints to ML predictions
 * Improves accuracy and logical consistency
 */
class BusinessRulesEngine {
    
    constructor() {
        // Prerequisite thresholds
        this.thresholds = {
            agama_studi_islam: 75,
            inggris_pend_inggris: 75,
            ekonomi_business: 70,
            stem_minimum: 70,
            science_health: 70
        };
    }

    /**
     * Apply all business rules to predictions
     * @param {Array} predictions - ML predictions with prodi and confidence
     * @param {Object} userProfile - User input features
     * @returns {Array} Adjusted predictions with warnings
     */
    applyRules(predictions, userProfile) {
        return predictions.map(pred => {
            const adjusted = {...pred};
            const validation = this.validatePrediction(pred.prodi, userProfile);
            
            // Adjust confidence based on rules
            adjusted.confidence = pred.confidence * validation.multiplier;
            adjusted.warnings = validation.warnings;
            adjusted.strengths = validation.strengths;
            adjusted.alternatives = validation.alternatives;
            
            return adjusted;
        }).sort((a, b) => b.confidence - a.confidence); // Re-rank
    }

    /**
     * Validate a specific prodi against user profile
     * @param {string} prodi - Program studi name
     * @param {Object} profile - User profile
     * @returns {Object} Validation result with multiplier and warnings
     */
    validatePrediction(prodi, profile) {
        const result = {
            multiplier: 1.0,
            warnings: [],
            strengths: [],
            alternatives: []
        };

        // Apply prodi-specific rules
        if (prodi.includes('Studi Islam')) {
            this._validateStudiIslam(profile, result);
        } else if (prodi.includes('Informatika')) {
            this._validateInformatika(profile, result);
        } else if (prodi.includes('Teknik Elektro')) {
            this._validateTeknikElektro(profile, result);
        } else if (prodi.includes('Farmasi')) {
            this._validateFarmasi(profile, result);
        } else if (prodi.includes('Teknologi Hasil Pertanian')) {
            this._validateTeknologiPertanian(profile, result);
        } else if (prodi.includes('Akuntansi')) {
            this._validateAkuntansi(profile, result);
        } else if (prodi.includes('Manajemen')) {
            this._validateManajemen(profile, result);
        } else if (prodi.includes('Pendidikan Bahasa Inggris')) {
            this._validatePendidikanInggris(profile, result);
        } else if (prodi.includes('PGSD')) {
            this._validatePGSD(profile, result);
        } else if (prodi.includes('Agribisnis')) {
            this._validateAgribisnis(profile, result);
        }

        return result;
    }

    // ===================================
    // PRODI-SPECIFIC VALIDATION RULES
    // ===================================

    _validateStudiIslam(p, result) {
        // Critical: Agama score
        if (p.agama < this.thresholds.agama_studi_islam) {
            result.multiplier *= 0.3;
            result.warnings.push('⚠️ Nilai Agama di bawah rekomendasi (minimal 75)');
            
            // Suggest alternatives if strong STEM
            if (p.mtk >= 80) {
                result.alternatives.push('S1 Informatika (nilai MTK tinggi)');
            }
        } else {
            result.strengths.push('✅ Nilai Agama mencukupi');
        }

        // Hafalan preference
        if (p.hafalan === 0 && p.agama < 80) {
            result.multiplier *= 0.6;
            result.warnings.push('ℹ️ Studi Islam biasanya memerlukan kemampuan hafalan');
        } else if (p.hafalan === 1) {
            result.strengths.push('✅ Memiliki kemampuan hafalan');
        }

        // Contradictory signals: Strong STEM but weak Islam
        if (p.mtk >= 85 && p.agama < 80) {
            result.multiplier *= 0.4;
            result.warnings.push('⚠️ Profil lebih cocok untuk program STEM');
            result.alternatives.push('S1 Informatika', 'S1 Teknik Elektro');
        }
    }

    _validateInformatika(p, result) {
        // STEM requirements
        const stemScore = (p.mtk + p.fisika) / 2;
        
        if (stemScore >= 80 && p.minat_teknik >= 4) {
            result.multiplier *= 1.3;
            result.strengths.push('✅ Nilai STEM dan minat teknik tinggi');
        } else if (stemScore < this.thresholds.stem_minimum) {
            result.multiplier *= 0.5;
            result.warnings.push('⚠️ Nilai Matematika/Fisika perlu lebih tinggi');
        }

        // Tech interest is crucial
        if (p.minat_teknik >= 4) {
            result.strengths.push('✅ Minat teknik sangat cocok');
        } else if (p.minat_teknik < 3) {
            result.multiplier *= 0.6;
            result.warnings.push('ℹ️ Minat teknik cukup rendah untuk Informatika');
        }

        // English helps
        if (p.inggris >= 80) {
            result.strengths.push('✅ Bahasa Inggris kuat (penting untuk programming)');
        }
    }

    _validateTeknikElektro(p, result) {
        // Strong physics + math needed
        if (p.fisika >= 80 && p.mtk >= 75) {
            result.multiplier *= 1.3;
            result.strengths.push('✅ Fisika dan Matematika kuat');
        } else if (p.fisika < this.thresholds.stem_minimum || p.mtk < this.thresholds.stem_minimum) {
            result.multiplier *= 0.5;
            result.warnings.push('⚠️ Teknik Elektro memerlukan Fisika dan MTK tinggi');
        }

        // Tech interest
        if (p.minat_teknik >= 4) {
            result.strengths.push('✅ Minat teknik sesuai');
        } else if (p.minat_teknik < 3) {
            result.multiplier *= 0.7;
        }
    }

    _validateFarmasi(p, result) {
        // Chemistry + Biology critical
        const scienceScore = (p.kimia + p.biologi) / 2;
        
        if (scienceScore >= 80 && p.minat_kesehatan >= 4) {
            result.multiplier *= 1.4;
            result.strengths.push('✅ Kimia, Biologi, dan minat kesehatan tinggi');
        } else if (p.kimia < this.thresholds.science_health && p.biologi < this.thresholds.science_health) {
            result.multiplier *= 0.3;
            result.warnings.push('⚠️ Farmasi memerlukan Kimia dan Biologi kuat');
        }

        // Health interest crucial
        if (p.minat_kesehatan >= 4) {
            result.strengths.push('✅ Minat kesehatan sesuai');
        } else if (p.minat_kesehatan < 3) {
            result.multiplier *= 0.6;
            result.warnings.push('ℹ️ Minat kesehatan cukup rendah');
        }
    }

    _validateTeknologiPertanian(p, result) {
        // Biology important
        if (p.biologi >= 75 && p.kimia >= 70) {
            result.multiplier *= 1.2;
            result.strengths.push('✅ Biologi dan Kimia mencukupi');
        } else if (p.biologi < 65) {
            result.multiplier *= 0.6;
            result.warnings.push('⚠️ Biologi perlu lebih tinggi');
        }

        // Can have mixed interests
        if (p.minat_kesehatan >= 3 || p.minat_teknik >= 3) {
            result.strengths.push('✅ Kombinasi minat sains cocok');
        }
    }

    _validateAkuntansi(p, result) {
        // Economics is key
        if (p.ekonomi >= 80 && p.minat_bisnis >= 4) {
            result.multiplier *= 1.3;
            result.strengths.push('✅ Ekonomi dan minat bisnis tinggi');
        } else if (p.ekonomi < this.thresholds.ekonomi_business) {
            result.multiplier *= 0.6;
            result.warnings.push('⚠️ Nilai Ekonomi penting untuk Akuntansi');
        }

        // Math helps
        if (p.mtk >= 80) {
            result.strengths.push('✅ Matematika kuat (bagus untuk Akuntansi)');
        }

        // Business interest
        if (p.minat_bisnis >= 4) {
            result.strengths.push('✅ Minat bisnis sesuai');
        }
    }

    _validateManajemen(p, result) {
        // Similar to Akuntansi but more flexible
        if (p.ekonomi >= 75 && p.minat_bisnis >= 4) {
            result.multiplier *= 1.2;
            result.strengths.push('✅ Ekonomi dan minat bisnis sesuai');
        } else if (p.ekonomi < this.thresholds.ekonomi_business) {
            result.multiplier *= 0.7;
            result.warnings.push('ℹ️ Ekonomi penting untuk Manajemen');
        }

        // Business interest crucial
        if (p.minat_bisnis >= 4) {
            result.strengths.push('✅ Minat bisnis tinggi');
        } else if (p.minat_bisnis < 3) {
            result.multiplier *= 0.6;
        }
    }

    _validatePendidikanInggris(p, result) {
        // English is critical
        if (p.inggris < this.thresholds.inggris_pend_inggris) {
            result.multiplier *= 0.4;
            result.warnings.push('⚠️ Pendidikan B. Inggris memerlukan nilai Inggris tinggi (min 75)');
        } else if (p.inggris >= 85) {
            result.multiplier *= 1.3;
            result.strengths.push('✅ Bahasa Inggris sangat baik');
        }

        // Education interest
        if (p.minat_pendidikan >= 4) {
            result.strengths.push('✅ Minat pendidikan tinggi');
        } else if (p.minat_pendidikan < 3) {
            result.multiplier *= 0.6;
            result.warnings.push('ℹ️ Minat pendidikan cukup rendah');
        }
    }

    _validatePGSD(p, result) {
        // Education interest is key
        if (p.minat_pendidikan < 3) {
            result.multiplier *= 0.5;
            result.warnings.push('⚠️ PGSD memerlukan minat pendidikan tinggi');
        } else if (p.minat_pendidikan >= 4) {
            result.multiplier *= 1.2;
            result.strengths.push('✅ Minat pendidikan sesuai');
        }

        // Balanced academic profile
        const avgScore = (p.mtk + p.inggris + p.agama) / 3;
        if (avgScore >= 75) {
            result.strengths.push('✅ Profil akademik seimbang');
        }
    }

    _validateAgribisnis(p, result) {
        // Biology + Economics combination
        if (p.biologi >= 70 && p.ekonomi >= 70) {
            result.multiplier *= 1.2;
            result.strengths.push('✅ Kombinasi Biologi dan Ekonomi baik');
        } else if (p.biologi < 65 && p.ekonomi < 65) {
            result.multiplier *= 0.6;
            result.warnings.push('⚠️ Agribisnis butuh Biologi dan Ekonomi');
        }

        // Business interest helps
        if (p.minat_bisnis >= 3) {
            result.strengths.push('✅ Minat bisnis mendukung');
        }
    }

    /**
     * Format confidence as percentage
     */
    formatConfidence(confidence) {
        return Math.round(Math.max(0, Math.min(100, confidence * 100)));
    }

    /**
     * Get confidence level label
     */
    getConfidenceLevel(confidence) {
        if (confidence >= 0.7) return 'Sangat Cocok';
        if (confidence >= 0.5) return 'Cocok';
        if (confidence >= 0.3) return 'Cukup Cocok';
        return 'Kurang Cocok';
    }
}

// ===================================
// EXPORT
// ===================================
if (typeof window !== 'undefined') {
    window.BusinessRulesEngine = BusinessRulesEngine;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = BusinessRulesEngine;
}
