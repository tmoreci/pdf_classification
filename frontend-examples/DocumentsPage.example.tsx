/**
 * Example Documents Page Component
 * Copy this to your frontend/src/pages/DocumentsPage.tsx
 */
import { useState } from 'react';
import { useDocuments } from '../hooks/useDocuments';
import { FileUpload } from '../components/common/FileUpload';
import { TrashIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

export default function DocumentsPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customDocId, setCustomDocId] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const { documents, totalCount, isLoading, upload, isUploading, delete: deleteDoc } = useDocuments();

  const handleUpload = () => {
    if (!selectedFile) return;
    upload({ file: selectedFile, docId: customDocId || undefined });
    setSelectedFile(null);
    setCustomDocId('');
  };

  const filteredDocuments = documents.filter(
    (doc) =>
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.summary.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Document Management</h1>
        <p className="mt-2 text-gray-600">
          Upload and manage your research paper collection
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Upload New Document</h2>

        <FileUpload
          onFileSelect={(file) => setSelectedFile(file)}
        />

        {selectedFile && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-blue-800">
                Selected: <span className="font-medium">{selectedFile.name}</span>
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Custom Document ID (optional)
              </label>
              <input
                type="text"
                value={customDocId}
                onChange={(e) => setCustomDocId(e.target.value)}
                placeholder="e.g., attention_2017"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 font-medium"
            >
              {isUploading ? 'Uploading...' : 'Upload Document'}
            </button>
          </div>
        )}
      </div>

      {/* Documents List */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">
            Documents ({totalCount})
          </h2>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documents..."
            className="px-4 py-2 border border-gray-300 rounded-lg w-64"
          />
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Loading documents...</p>
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-12 h-12 mx-auto text-gray-400" />
            <p className="mt-2 text-gray-600">
              {searchQuery ? 'No documents found' : 'No documents uploaded yet'}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredDocuments.map((doc) => (
              <div
                key={doc.doc_id}
                className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{doc.title}</h3>
                    <p className="text-sm text-gray-500 mt-1">ID: {doc.doc_id}</p>
                    <p className="mt-2 text-gray-600 line-clamp-2">{doc.summary}</p>
                  </div>
                  <button
                    onClick={() => {
                      if (window.confirm(`Delete "${doc.title}"?`)) {
                        deleteDoc(doc.doc_id);
                      }
                    }}
                    className="ml-4 p-2 text-red-600 hover:bg-red-50 rounded-lg"
                    title="Delete document"
                  >
                    <TrashIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
