import React from 'react';
import NavBar from './Navbar';
import Footer from './Footer';

function Layout({ children }) {
  return (
    <div className="flex flex-col min-h-screen">
      <NavBar />
      <main className="flex-grow">
        {children}
      </main>
      <Footer />
    </div>
  );
}

export default Layout;