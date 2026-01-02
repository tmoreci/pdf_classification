/**
 * Example Search Page Component
 * Copy this to your frontend/src/pages/SearchPage.tsx
 */
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { searchApi } from '../api/endpoints';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import type { SearchResponse } from '../types/api';

type SearchMode = 'hybrid' | 'vector' | 'keyword';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState<SearchMode>('hybrid');
  const [nResults, setNResults] = useState(5);
  const [vectorWeight, setVectorWeight] = useState(0.7);
  const [results, setResults] = useState<SearchResponse | null>(null);

  const { mutate: search, isPending } = useMutation({
    mutationFn: async () => {
      const baseRequest = { query, n_results: nResults };

      switch (searchMode) {
        case 'vector':
          return searchApi.vector(baseRequest);
        case 'keyword':
          return searchApi.keyword(baseRequest);
        case 'hybrid':
          return searchApi.hybrid({
            ...baseRequest,
            vector_weight: vectorWeight,
            keyword_weight: 1 - vectorWeight,
          });
      }
    },
    onSuccess: (data) => {
      setResults(data);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    search();
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Search Documents</h1>
        <p className="mt-2 text-gray-600">
          Search your document collection using semantic or keyword search
        </p>
      </div>

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          {/* Search Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
            </label>
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., transformer architecture for NLP"
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
              <MagnifyingGlassIcon className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
            </div>
          </div>

          {/* Search Mode */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Mode
            </label>
            <div className="flex gap-2">
              {(['hybrid', 'vector', 'keyword'] as SearchMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setSearchMode(mode)}
                  className={`
                    px-4 py-2 rounded-lg font-medium capitalize
                    ${searchMode === mode
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }
                  `}
                >
                  {mode}
                </button>
              ))}
            </div>
            <p className="mt-2 text-sm text-gray-500">
              {searchMode === 'hybrid' && 'Combines semantic and keyword search'}
              {searchMode === 'vector' && 'Semantic similarity using embeddings'}
              {searchMode === 'keyword' && 'Traditional BM25 keyword matching'}
            </p>
          </div>

          {/* Hybrid Settings */}
          {searchMode === 'hybrid' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Vector Weight: {(vectorWeight * 100).toFixed(0)}%
                (Keyword: {((1 - vectorWeight) * 100).toFixed(0)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={vectorWeight}
                onChange={(e) => setVectorWeight(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          {/* Number of Results */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Results: {nResults}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value={nResults}
              onChange={(e) => setNResults(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isPending || !query.trim()}
            className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 font-medium"
          >
            {isPending ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Results */}
      {results && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">
            Results for "{results.query}" ({results.total_results})
          </h2>

          {results.results.length === 0 ? (
            <p className="text-gray-600 text-center py-8">No results found</p>
          ) : (
            <div className="space-y-4">
              {results.results.map((result, index) => (
                <div
                  key={`${result.doc_id}-${index}`}
                  className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-500">
                          #{index + 1}
                        </span>
                        <h3 className="font-semibold text-gray-900">
                          {result.title}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">
                        ID: {result.doc_id}
                      </p>
                      <p className="mt-2 text-gray-600">{result.content}</p>
                    </div>
                    {result.score !== null && result.score !== undefined && (
                      <div className="ml-4 text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {(result.score * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">similarity</div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
