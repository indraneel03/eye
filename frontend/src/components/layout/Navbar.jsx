import React from 'react';
import { Link } from 'react-router-dom';
import { Eye, Info, Home } from 'lucide-react';

const NavBar = () => {
  return (
    <nav className="bg-blue-900 text-white">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center">
            <Eye className="w-8 h-8 mr-2" />
            <span className="font-bold text-xl">OCT Analysis</span>
          </Link>
          <div className="flex space-x-4">
            <Link to="/" className="flex items-center px-3 py-2 rounded-md hover:bg-blue-800">
              <Home className="w-4 h-4 mr-1" />
              Home
            </Link>
            <Link to="/classify" className="flex items-center px-3 py-2 rounded-md hover:bg-blue-800">
              <Eye className="w-4 h-4 mr-1" />
              Classify
            </Link>
            <Link to="/about" className="flex items-center px-3 py-2 rounded-md hover:bg-blue-800">
              <Info className="w-4 h-4 mr-1" />
              About
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;