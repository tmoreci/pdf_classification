/**
 * Example Q&A Page Component
 * Copy this to your frontend/src/pages/QAPage.tsx
 */
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { qaApi } from '../api/endpoints';
import { FileUpload } from '../components/common/FileUpload';
import type { QAResponse } from '../types/api';

export default function QAPage() {
  const [apiKey, setApiKey] = useState('');
  const [question, setQuestion] = useState('');
  const [documentContent, setDocumentContent] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [answer, setAnswer] = useState<QAResponse | null>(null);

  const { mutate: askQuestion, isPending, error } = useMutation({
    mutationFn: qaApi.answer,
    onSuccess: (data) => {
      setAnswer(data);
    },
  });

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    // Extract text from PDF (you'll need a PDF parsing library)
    const text = await extractTextFromPDF(file);
    setDocumentContent(text);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!apiKey || !question || !documentContent) return;

    askQuestion({
      question,
      document_content: documentContent,
      api_key: apiKey,
      llm_provider: 'gemini',
      model: 'gemini-2.0-flash-exp',
      temperature: 0.1,
    });
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Q&A Interface</h1>
        <p className="mt-2 text-gray-600">
          Upload a PDF and ask questions about its content
        </p>
      </div>

      {/* API Key Input */}
      <div className="bg-white rounded-lg shadow p-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Gemini API Key
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Enter your Gemini API key"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <p className="mt-2 text-sm text-gray-500">
          Get your API key from{' '}
          <a
            href="https://ai.google.dev/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline"
          >
            Google AI Studio
          </a>
        </p>
      </div>

      {/* PDF Upload */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Upload Document</h2>
        <FileUpload onFileSelect={handleFileSelect} />
        {selectedFile && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-green-800">
              ✓ Loaded: <span className="font-medium">{selectedFile.name}</span>
            </p>
          </div>
        )}
      </div>

      {/* Question Form */}
      <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Ask a Question</h2>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What is the main contribution of this paper?"
          rows={3}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <button
          type="submit"
          disabled={isPending || !apiKey || !documentContent || !question}
          className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {isPending ? 'Generating Answer...' : 'Ask Question'}
        </button>
      </form>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">
            Error: {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      )}

      {/* Answer Display */}
      {answer && (
        <div className="space-y-6">
          {/* Main Answer */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-2">Answer</h2>
            <p className="text-gray-800 whitespace-pre-wrap">{answer.answer}</p>
          </div>

          {/* Thinking Process (Optional) */}
          {answer.thinking && (
            <details className="bg-gray-50 rounded-lg p-4">
              <summary className="cursor-pointer font-medium text-gray-700">
                View Thinking Process
              </summary>
              <p className="mt-2 text-gray-600 text-sm whitespace-pre-wrap">
                {answer.thinking}
              </p>
            </details>
          )}

          {/* Citations */}
          {answer.cited_documents.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">
                Cited Sources ({answer.cited_documents.length})
              </h2>
              <div className="space-y-4">
                {answer.cited_documents.map((doc, index) => (
                  <div
                    key={doc.doc_id}
                    className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
                  >
                    <h3 className="font-medium text-gray-900">
                      [{index + 1}] {doc.title}
                    </h3>
                    <p className="mt-2 text-gray-600 text-sm line-clamp-3">
                      {doc.content}
                    </p>
                    <span className="text-xs text-gray-500 mt-2 block">
                      Document ID: {doc.doc_id}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Helper function - you'll need to implement PDF text extraction
// Option 1: Use pdf.js (client-side)
// Option 2: Use the API's /pdf/parse-full-text endpoint
async function extractTextFromPDF(file: File): Promise<string> {
  // Placeholder - implement using pdf.js or call the API endpoint
  // For simplicity, you could call your API's /pdf/parse-full-text endpoint:

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/pdf/parse-full-text', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  return data.content;
}
