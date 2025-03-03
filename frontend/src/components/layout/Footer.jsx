import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-blue-900 text-white py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-bold mb-4">OCT Analysis</h3>
            <p className="text-sm">Advanced retinal disease detection powered by AI</p>
          </div>
          <div>
            <h3 className="text-lg font-bold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><Link to="/">Home</Link></li>
              <li><Link to="/classify">Classify</Link></li>
              <li><Link to="/about">About</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-bold mb-4">Contact</h3>
            <p className="text-sm">Email: indraneelkalva@gmail.com</p>
            <p className="text-sm">Phone: 7730841030</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;