import React from 'react';

interface LayoutProps {
  children?: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="mx-auto flex flex-col bg-custom-pink">
      <header className="sticky top-0 z-40 bg-custom-blue px-4 md:px-8 py-2 md:py-4 flex items-center justify-between">
        <div className="flex items-center">
          <a href="https://e-albania.al/" className="hover:text-slate-600 cursor-pointer text-white  text-xl md:text-2xl flex items-center">
            <img src="/logo.svg" alt="Logo" className="h-8 w-auto mr-2" />
            e-Albania 
          </a>
        </div>
      </header>
      <div className="pt-8">
        <main className="flex w-full flex-1 flex-col overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
