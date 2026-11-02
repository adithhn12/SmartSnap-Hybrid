import React, { useState, useRef } from 'react';
import {
  Camera,
  Upload,
  Image as ImageIcon,
  AlertCircle,
  ArrowRight,
  X
} from 'lucide-react';
import { analyzePhoto } from '../services/api';

const CATEGORIES = [
  { id: 'General Photography', label: 'General Photography' },
  { id: 'Portrait', label: 'Portrait' },
  { id: 'Group Photograph', label: 'Group Photograph' },
  { id: 'Landscape', label: 'Landscape' },
  { id: 'Document', label: 'Document' },
];

export default function AnalysisPage({ onAnalysisComplete }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [category, setCategory] = useState('General Photography');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Separate refs for standard file picker and direct native camera capture
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;

    // Check valid image types (JPEG, JPG, PNG)
    const validMimeTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const validExtensions = /\.(jpg|jpeg|png)$/i;

    const isValidMime = validMimeTypes.includes(selectedFile.type);
    const isValidExt = validExtensions.test(selectedFile.name || '');

    if (!isValidMime && !isValidExt) {
      setError('Please select or capture a valid JPG, JPEG, or PNG photograph.');
      return;
    }

    // Revoke previous object URL to avoid memory leaks
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setError(null);
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  const handleClearPhoto = (e) => {
    if (e) e.stopPropagation();
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setFile(null);
    setPreviewUrl(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please take a photograph or select one to analyse.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await analyzePhoto(file, category);
      onAnalysisComplete(result);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please check that the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return '';
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div>
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '1.65rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px', letterSpacing: '-0.02em' }}>
          Photograph Upload & Analysis
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          A Random Forest estimates perceptual MOS from image pixels; Mamdani fuzzy inference combines that estimate with independent face and composition context.
        </p>
      </div>

      {error && (
        <div style={{ background: 'var(--retake-bg)', border: '1px solid var(--retake-border)', borderRadius: '6px', padding: '12px 16px', marginBottom: '18px', color: 'var(--retake-color)', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.88rem' }}>
          <AlertCircle size={16} color="var(--retake-color)" style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}

      {/* Hidden File Inputs */}
      {/* 1. Direct native camera input (prefers rear-facing camera on mobile via capture="environment") */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/*"
        capture="environment"
        style={{ display: 'none' }}
        onChange={(e) => handleFileChange(e.target.files[0])}
      />

      {/* 2. Standard gallery / file manager picker */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        style={{ display: 'none' }}
        onChange={(e) => handleFileChange(e.target.files[0])}
      />

      <form onSubmit={handleSubmit}>
        <div className="grid-2 analysis-grid">
          {/* 1. Left Column: Photograph Selection & Preview */}
          <div className="card">
            <h2 className="card-title">Photograph Selection</h2>
            <p className="card-subtitle">
              Capture directly using your phone's camera or select an existing photo
            </p>

            {/* Dropzone Area */}
            <div
              className={`dropzone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => {
                // If user clicks on the empty dropzone area, open file picker
                if (!file) fileInputRef.current?.click();
              }}
            >
              <ImageIcon className="dropzone-icon" />
              <p style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-primary)', marginBottom: '4px', wordBreak: 'break-word' }}>
                {file ? file.name : 'Take a photo or choose from gallery'}
              </p>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Supports JPG, JPEG, and PNG formats &bull; Direct camera capture available
              </p>

              {/* Action Buttons */}
              <div className="dropzone-actions" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => cameraInputRef.current?.click()}
                  style={{ minHeight: '42px', padding: '10px 16px' }}
                >
                  <Camera size={16} />
                  <span>Take Photo</span>
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => fileInputRef.current?.click()}
                  style={{ minHeight: '42px', padding: '10px 16px' }}
                >
                  <Upload size={16} />
                  <span>Choose from Gallery</span>
                </button>
              </div>
            </div>

            {/* Photograph Preview Area */}
            {previewUrl && (
              <div style={{ marginTop: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span className="form-label" style={{ marginBottom: 0 }}>
                    Selected Photograph Preview
                  </span>
                  {file?.size && (
                    <span style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                      {formatFileSize(file.size)}
                    </span>
                  )}
                </div>

                <div style={{ borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border)', maxHeight: '360px', background: 'var(--bg-subtle-gray)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6px' }}>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{ maxWidth: '100%', maxHeight: '348px', width: 'auto', height: 'auto', objectFit: 'contain', borderRadius: '4px' }}
                  />
                </div>

                {/* Replacement & Clear Actions */}
                <div className="preview-actions">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => cameraInputRef.current?.click()}
                    style={{ fontSize: '0.82rem', padding: '6px 12px' }}
                  >
                    <Camera size={14} />
                    <span>Retake Photo</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => fileInputRef.current?.click()}
                    style={{ fontSize: '0.82rem', padding: '6px 12px' }}
                  >
                    <Upload size={14} />
                    <span>Choose Different</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={handleClearPhoto}
                    style={{ fontSize: '0.82rem', padding: '6px 12px' }}
                  >
                    <X size={14} />
                    <span>Remove</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: 
              Order on Mobile & Desktop:
              1. Category Selection
              2. Analyse Photograph Button (Immediately below Category!)
              3. Computer Vision Feature Extraction (Reference card below Button!)
          */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {/* 2. Simplified Category Selection */}
            <div className="card">
              <h2 className="card-title">Category</h2>
              <p className="card-subtitle" style={{ marginBottom: '12px' }}>
                Select photograph category to calibrate feature priorities
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {CATEGORIES.map((cat) => {
                  const isSelected = category === cat.id;
                  return (
                    <label
                      key={cat.id}
                      className={`category-option-row ${isSelected ? 'selected' : ''}`}
                    >
                      <input
                        type="radio"
                        name="category"
                        value={cat.id}
                        checked={isSelected}
                        onChange={(e) => setCategory(e.target.value)}
                        style={{ accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
                      />
                      <span className="category-option-label">
                        {cat.label}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* 3. Primary Action: Analyse Photograph Button (Immediately accessible!) */}
            <button
              type="submit"
              className="btn btn-primary submit-analysis-btn"
              disabled={!file || isLoading}
            >
              {isLoading ? (
                <span>Extracting Features & Evaluating FIS...</span>
              ) : (
                <>
                  <span>Analyse Photograph</span>
                  <ArrowRight size={16} />
                </>
              )}
            </button>

            {/* 4. Computer Vision Feature Extraction (Positioned below the action button) */}
            <div className="card" style={{ backgroundColor: 'var(--bg-subtle-gray)' }}>
              <div style={{ fontWeight: 600, fontSize: '0.86rem', color: 'var(--text-primary)', marginBottom: '4px' }}>
                Computer Vision Feature Extraction
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '10px', lineHeight: '1.45' }}>
                OpenCV extracts pixel-only features for the trained Random Forest, then the existing CV pipeline measures independent context for the Mamdani system:
              </p>
              <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '5px', lineHeight: '1.45' }}>
                <li><strong style={{ color: 'var(--text-primary)' }}>Random Forest features:</strong> Brightness, contrast, colorfulness, global sharpness, and noise.</li>
                <li><strong style={{ color: 'var(--text-primary)' }}>Fuzzy model input:</strong> Predicted KonIQ MOS on its native 1&ndash;5 scale.</li>
                <li><strong style={{ color: 'var(--text-primary)' }}>Independent CV context:</strong> Existing face visibility and composition estimates; model-input proxies are not duplicated in the hybrid FIS.</li>
              </ul>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
