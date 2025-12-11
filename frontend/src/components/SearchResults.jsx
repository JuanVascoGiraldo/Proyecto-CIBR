import './SearchResults.css'

function SearchResults({ results, apiUrl }) {
  if (!results || !results.results) return null

  const getImageUrl = (imagePath) => {
    // Convertir path de Windows a URL relativa del frontend
    const normalizedPath = imagePath.replace(/\\/g, '/')
    // Servir desde el public del frontend
    return `/${normalizedPath}`
  }

  return (
    <div className="search-results">
      <div className="results-header">
        <h2>Resultados de Búsqueda</h2>
        <div className="search-info">
          <span className="info-badge">
            <strong>Imagen:</strong> {results.query_image}
          </span>
          <span className="info-badge">
            <strong>Extractor:</strong> {results.extractor}
          </span>
          <span className="info-badge">
            <strong>Índice:</strong> {results.index_type}
          </span>
          <span className="info-badge">
            <strong>Dimensión:</strong> {results.feature_dimension}
          </span>
        </div>
      </div>

      <div className="results-grid">
        {results.results.map((result) => (
          <div key={result.rank} className="result-card">
            <div className="result-rank">#{result.rank}</div>
            
            <div className="result-image-container">
              <img
                src={getImageUrl(result.image_path)}
                alt={result.filename}
                onError={(e) => {
                  e.target.onerror = null
                  e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23ddd"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%23999"%3ENo disponible%3C/text%3E%3C/svg%3E'
                }}
              />
            </div>

            <div className="result-info">
              <h3 className="result-filename">{result.filename}</h3>
              
              <div className="result-details">
                <div className="detail-row">
                  <span className="detail-label">Categoría:</span>
                  <span className={`category-badge ${result.category}`}>
                    {result.category}
                  </span>
                </div>
                
                <div className="detail-row">
                  <span className="detail-label">Conjunto:</span>
                  <span className={`split-badge ${result.split}`}>
                    {result.split}
                  </span>
                </div>
                
                <div className="detail-row">
                  <span className="detail-label">Distancia:</span>
                  <span className="distance-value">
                    {result.distance.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="similarity-bar">
                <div
                  className="similarity-fill"
                  style={{
                    width: `${Math.max(0, 100 - result.distance * 100)}%`
                  }}
                ></div>
              </div>
              <p className="similarity-text">
                Similitud: {Math.max(0, 100 - result.distance * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SearchResults
