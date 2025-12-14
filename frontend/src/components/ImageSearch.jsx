import { useState } from 'react'
import './ImageSearch.css'

function ImageSearch({ extractors, indices, onSearch, loading }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [selectedExtractor, setSelectedExtractor] = useState('ResNet50')
  const [selectedIndex, setSelectedIndex] = useState('flat')
  const [numResults, setNumResults] = useState(10)

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
      
      // Crear preview
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    
    if (!selectedFile) {
      alert('Por favor selecciona una imagen')
      return
    }

    onSearch(selectedFile, selectedExtractor, selectedIndex, numResults)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file)
      
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <div className="image-search">
      <form onSubmit={handleSubmit}>
        <div className="upload-section">
          <div
            className={`dropzone ${selectedFile ? 'has-image' : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            {previewUrl ? (
              <div className="preview">
                <img src={previewUrl} alt="Preview" />
                <p className="filename">{selectedFile?.name}</p>
              </div>
            ) : (
              <div className="dropzone-placeholder">
                <div className="upload-icon">📁</div>
                <p>Arrastra una imagen aquí</p>
                <p className="or">o</p>
                <button type="button" className="select-btn">
                  Seleccionar archivo
                </button>
              </div>
            )}
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>
        </div>

        <div className="options-section">
          <div className="form-group">
            <label htmlFor="extractor">Extractor de Características:</label>
            <select
              id="extractor"
              value={selectedExtractor}
              onChange={(e) => setSelectedExtractor(e.target.value)}
              disabled={loading}
            >
              {extractors.map((ext) => (
                <option key={ext} value={ext}>
                  {ext}
                </option>
              ))}
            </select>
            <small className="help-text">
              {selectedExtractor === 'ResNet50' && 'CNN profunda - Mejor similitud semántica'}
              {selectedExtractor === 'VGG16' && 'CNN compacta - Balance velocidad/precisión'}
              {selectedExtractor === 'ColorTexture' && 'Color + Textura - Muy rápido'}
              {selectedExtractor === 'HOG' && 'Detecta formas y bordes'}
              {selectedExtractor === 'ColorShape' && 'Color + Forma - Más compacto'}
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="index">Tipo de Índice:</label>
            <select
              id="index"
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(e.target.value)}
              disabled={loading}
            >
              {indices.map((idx) => (
                <option key={idx} value={idx}>
                  {idx.toUpperCase()}
                </option>
              ))}
            </select>
            <small className="help-text">
              {selectedIndex === 'flat' && 'Búsqueda exacta - 100% precisión'}
              {selectedIndex === 'ivf' && 'Rápida con clustering'}
              {selectedIndex === 'ivfpq' && 'Muy rápida con compresión'}
              {selectedIndex === 'hnsw' && 'Alta precisión y velocidad'}
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="k">Número de Resultados:</label>
            <input
              id="k"
              type="range"
              min="1"
              max="20"
              value={numResults}
              onChange={(e) => setNumResults(parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="range-value">{numResults}</span>
          </div>
        </div>

        <button
          type="submit"
          className="search-btn"
          disabled={!selectedFile || loading}
        >
          {loading ? 'Buscando...' : ' Buscar Imágenes Similares'}
        </button>
      </form>
    </div>
  )
}

export default ImageSearch
