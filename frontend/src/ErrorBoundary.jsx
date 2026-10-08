import React from 'react';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("React Error Caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 20, color: 'red', fontFamily: 'sans-serif' }}>
          <h2>Frontend Crashed!</h2>
          <pre style={{ background: '#f5f5f5', padding: 10, borderRadius: 5 }}>
            {this.state.error?.toString()}
          </pre>
          <p>Please report this error.</p>
        </div>
      );
    }
    return this.props.children;
  }
}
