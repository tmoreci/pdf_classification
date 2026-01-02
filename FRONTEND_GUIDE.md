# Frontend Development Guide

This guide will help you build a modern frontend for the PDF Q&A API.

## Quick Start with React + TypeScript + Vite

### 1. Create the Frontend Project

```bash
# From the pdf_qa directory
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

### 2. Install Essential Dependencies

```bash
# API client and state management
npm install axios @tanstack/react-query

# Routing
npm install react-router-dom

# File upload
npm install react-dropzone

# UI components and styling
npm install tailwindcss postcss autoprefixer -D
npm install @headlessui/react @heroicons/react
npm install clsx

# Charts for topic visualization
npm install recharts

# Form handling
npm install react-hook-form zod @hookform/resolvers
```

### 3. Initialize Tailwind CSS

```bash
npx tailwindcss init -p
```

Update `tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

Add to `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Recommended Project Structure

```
frontend/
├── public/
├── src/
│   ├── api/
│   │   ├── client.ts           # Axios instance
│   │   └── endpoints.ts        # API endpoint functions
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── FileUpload.tsx
│   │   │   ├── Loading.tsx
│   │   │   └── ErrorMessage.tsx
│   │   ├── documents/
│   │   │   ├── DocumentList.tsx
│   │   │   ├── DocumentCard.tsx
│   │   │   └── DocumentUpload.tsx
│   │   ├── search/
│   │   │   ├── SearchBar.tsx
│   │   │   └── SearchResults.tsx
│   │   ├── qa/
│   │   │   ├── QuestionInput.tsx
│   │   │   ├── AnswerDisplay.tsx
│   │   │   └── CitationCard.tsx
│   │   └── topics/
│   │       ├── TopicChart.tsx
│   │       └── TopicList.tsx
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   ├── DocumentsPage.tsx
│   │   ├── SearchPage.tsx
│   │   ├── QAPage.tsx
│   │   └── TopicsPage.tsx
│   ├── hooks/
│   │   ├── useDocuments.ts
│   │   ├── useSearch.ts
│   │   ├── useQA.ts
│   │   └── useTopics.ts
│   ├── types/
│   │   └── api.ts              # TypeScript types from API
│   ├── utils/
│   │   └── formatting.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Essential Code Templates

### API Client Setup (`src/api/client.ts`)

```typescript
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.error || 'An error occurred';
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);
```

### API Endpoints (`src/api/endpoints.ts`)

```typescript
import { apiClient } from './client';
import type {
  DocumentUploadResponse,
  DocumentListResponse,
  SearchRequest,
  SearchResponse,
  QARequest,
  QAResponse,
  TopicModelingResponse,
} from '../types/api';

// Document Management
export const documentsApi = {
  upload: async (file: File, docId?: string): Promise<DocumentUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (docId) formData.append('doc_id', docId);

    const { data } = await apiClient.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  list: async (): Promise<DocumentListResponse> => {
    const { data } = await apiClient.get('/documents/list');
    return data;
  },

  get: async (docId: string) => {
    const { data } = await apiClient.get(`/documents/${docId}`);
    return data;
  },

  delete: async (docId: string) => {
    const { data } = await apiClient.delete(`/documents/${docId}`);
    return data;
  },
};

// Search
export const searchApi = {
  hybrid: async (request: SearchRequest & { vector_weight?: number; keyword_weight?: number }): Promise<SearchResponse> => {
    const { data } = await apiClient.post('/search/hybrid', request);
    return data;
  },

  vector: async (request: SearchRequest): Promise<SearchResponse> => {
    const { data } = await apiClient.post('/search/vector', request);
    return data;
  },

  keyword: async (request: SearchRequest): Promise<SearchResponse> => {
    const { data } = await apiClient.post('/search/keyword', request);
    return data;
  },
};

// Q&A
export const qaApi = {
  answer: async (request: QARequest): Promise<QAResponse> => {
    const { data } = await apiClient.post('/qa/answer', request);
    return data;
  },
};

// Topics
export const topicsApi = {
  analyzeUnsupervised: async (request: {
    n_topics?: number;
    min_topic_size: number;
    api_key: string;
  }): Promise<TopicModelingResponse> => {
    const { data } = await apiClient.post('/topics/analyze-unsupervised', request);
    return data;
  },

  analyzeZeroShot: async (request: {
    topics: string[];
    min_topic_size: number;
    api_key: string;
  }): Promise<TopicModelingResponse> => {
    const { data } = await apiClient.post('/topics/analyze-zeroshot', request);
    return data;
  },
};
```

### TypeScript Types (`src/types/api.ts`)

```typescript
// Match the API schemas exactly
export interface DocumentUploadResponse {
  doc_id: string;
  title: string;
  message: string;
}

export interface DocumentInfo {
  doc_id: string;
  title: string;
  summary: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total_count: number;
}

export interface SearchRequest {
  query: string;
  n_results?: number;
}

export interface SearchResult {
  doc_id: string;
  title: string;
  content: string;
  score?: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
}

export interface QARequest {
  question: string;
  document_content: string;
  api_key: string;
  model?: string;
  temperature?: number;
  llm_provider?: string;
}

export interface CitedDocument {
  doc_id: string;
  title: string;
  content: string;
}

export interface QAResponse {
  question: string;
  answer: string;
  cited_documents: CitedDocument[];
  thinking?: string;
}

export interface TopicInfo {
  topic_id: number;
  topic_label: string;
  count: number;
  keywords: string[];
}

export interface DocumentTopicInfo {
  doc_id: string;
  title: string;
  topic_id: number;
  topic_label: string;
  probability?: number;
}

export interface TopicModelingResponse {
  topics: TopicInfo[];
  document_topics: DocumentTopicInfo[];
  total_documents: number;
  total_topics: number;
}
```

### React Query Hook Example (`src/hooks/useDocuments.ts`)

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentsApi } from '../api/endpoints';

export const useDocuments = () => {
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
  });

  const uploadMutation = useMutation({
    mutationFn: ({ file, docId }: { file: File; docId?: string }) =>
      documentsApi.upload(file, docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: documentsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  return {
    documents: data?.documents || [],
    totalCount: data?.total_count || 0,
    isLoading,
    error,
    upload: uploadMutation.mutate,
    isUploading: uploadMutation.isPending,
    delete: deleteMutation.mutate,
    isDeleting: deleteMutation.isPending,
  };
};
```

### File Upload Component (`src/components/common/FileUpload.tsx`)

```typescript
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon } from '@heroicons/react/24/outline';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  accept = '.pdf',
  maxSize = 10485760, // 10MB
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxSize,
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
        transition-colors duration-200
        ${isDragActive
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-300 hover:border-gray-400'
        }
      `}
    >
      <input {...getInputProps()} />
      <CloudArrowUpIcon className="w-12 h-12 mx-auto mb-4 text-gray-400" />
      {isDragActive ? (
        <p className="text-blue-600">Drop the PDF here...</p>
      ) : (
        <div>
          <p className="text-gray-600">
            Drag and drop a PDF file here, or click to select
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Maximum file size: {(maxSize / 1048576).toFixed(0)}MB
          </p>
        </div>
      )}
    </div>
  );
};
```

### Main App Setup (`src/App.tsx`)

```typescript
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import HomePage from './pages/HomePage';
import DocumentsPage from './pages/DocumentsPage';
import SearchPage from './pages/SearchPage';
import QAPage from './pages/QAPage';
import TopicsPage from './pages/TopicsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          {/* Navigation */}
          <nav className="bg-white shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex space-x-8">
                  <Link to="/" className="inline-flex items-center px-1 pt-1 text-gray-900">
                    Home
                  </Link>
                  <Link to="/documents" className="inline-flex items-center px-1 pt-1 text-gray-500 hover:text-gray-900">
                    Documents
                  </Link>
                  <Link to="/search" className="inline-flex items-center px-1 pt-1 text-gray-500 hover:text-gray-900">
                    Search
                  </Link>
                  <Link to="/qa" className="inline-flex items-center px-1 pt-1 text-gray-500 hover:text-gray-900">
                    Q&A
                  </Link>
                  <Link to="/topics" className="inline-flex items-center px-1 pt-1 text-gray-500 hover:text-gray-900">
                    Topics
                  </Link>
                </div>
              </div>
            </div>
          </nav>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/qa" element={<QAPage />} />
              <Route path="/topics" element={<TopicsPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
```

### Environment Variables (`.env`)

```bash
VITE_API_URL=http://localhost:8000
```

## Development Workflow

### 1. Start the API Server

```bash
# In the pdf_qa directory
python run_api.py
```

### 2. Start the Frontend Dev Server

```bash
# In the frontend directory
npm run dev
```

### 3. Open in Browser

- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

## Key Features to Implement First

### Phase 1: Core Functionality (Week 1)
1. ✅ **API Client Setup** - Connect to backend
2. ✅ **Document Upload** - Upload PDFs
3. ✅ **Document List** - View indexed documents
4. ✅ **Basic Search** - Search functionality

### Phase 2: Q&A Interface (Week 2)
1. ✅ **Question Input** - Ask questions
2. ✅ **Answer Display** - Show answers with formatting
3. ✅ **Citations** - Display cited sources
4. ✅ **API Key Management** - Store Gemini/Cohere keys

### Phase 3: Advanced Features (Week 3)
1. ✅ **Topic Modeling** - Visualize topics
2. ✅ **Advanced Search** - Filters and options
3. ✅ **Document Management** - Edit, delete documents
4. ✅ **Settings** - User preferences

### Phase 4: Polish (Week 4)
1. ✅ **Error Handling** - Better UX for errors
2. ✅ **Loading States** - Skeletons and spinners
3. ✅ **Responsive Design** - Mobile optimization
4. ✅ **Performance** - Lazy loading, caching

## Alternative: Component Libraries

If you want to move faster, consider using a component library:

### Option 1: shadcn/ui (Recommended)
```bash
npx shadcn-ui@latest init
npx shadcn-ui@latest add button input card dialog
```

### Option 2: Chakra UI
```bash
npm install @chakra-ui/react @emotion/react @emotion/styled framer-motion
```

### Option 3: Material-UI (MUI)
```bash
npm install @mui/material @emotion/react @emotion/styled
```

## Deployment

### Build for Production

```bash
npm run build
```

### Deploy Options
- **Vercel**: `vercel --prod`
- **Netlify**: Drag & drop `dist/` folder
- **AWS S3 + CloudFront**: Upload `dist/` to S3
- **Docker**: Serve with nginx

## Next Steps

1. **Choose your stack** (I recommend React + TypeScript + Vite)
2. **Set up the project** with the templates above
3. **Implement Phase 1** (API client + document management)
4. **Test with your API** running locally
5. **Iterate and expand** with more features

## Resources

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TanStack Query Docs](https://tanstack.com/query/latest)
- [Tailwind CSS](https://tailwindcss.com/)
- [Vite Guide](https://vitejs.dev/guide/)

Happy coding! 🚀
