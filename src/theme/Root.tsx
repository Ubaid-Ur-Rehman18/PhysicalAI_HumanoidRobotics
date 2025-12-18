import React from 'react';
import ChatBot from '@site/src/components/ChatBot';

/**
 * Root component that wraps the entire Docusaurus application.
 * The ChatBot will appear on every page of the documentation.
 */
export default function Root({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <>
      {children}
      <ChatBot />
    </>
  );
}
