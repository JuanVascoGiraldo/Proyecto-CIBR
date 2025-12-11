import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import ImageSearch from './components/ImageSearch'
import SearchResults from './components/SearchResults'

const API_URL = 'http://localhost:8000'

function App() {
  const [extractors, setExtractors] = useState([])
  const [indices, setIndices] = useState([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Cargar extractores e índices al iniciar
    loadExtractorsAndIndices()
  }, [])

  const loadExtractorsAndIndices = async () => {
    try {
      const [extractorsRes, indicesRes] = await Promise.all([
        axios.get(`${API_URL}/extractors`),
        axios.get(`${API_URL}/indices`)
      ])
      
      setExtractors(Object.keys(extractorsRes.data.extractors))
      setIndices(Object.keys(indicesRes.data.indices))
    } catch (err) {
      console.error('Error cargando configuración:', err)
      setError('No se pudo conectar con la API. Asegúrate de que esté ejecutándose.')
    }
  }

  const handleSearch = async (file, extractor, indexType, k) => {
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(
        `${API_URL}/search`,
        formData,
        {
          params: {
            extractor: extractor,
            index_type: indexType,
            k: k
          },
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )

      setResults(response.data)
    } catch (err) {
      console.error('Error en búsqueda:', err)
      setError(err.response?.data?.detail || 'Error al buscar imágenes similares')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎵 Sistema de Recuperación de Imágenes</h1>
        <p>Busca instrumentos musicales similares usando IA</p>
      </header>

      <main className="App-main">
        <ImageSearch
          extractors={extractors}
          indices={indices}
          onSearch={handleSearch}
          loading={loading}
        />

        {error && (
          <div className="error-message">
            <p>⚠️ {error}</p>
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Buscando imágenes similares...</p>
          </div>
        )}

        {results && !loading && (
          <SearchResults results={results} apiUrl={API_URL} />
        )}
      </main>

      <footer className="App-footer">
        <p>Proyecto CIBR - Sistema de Búsqueda de Imágenes con FAISS</p>
      </footer>
    </div>
  )
}

export default App
