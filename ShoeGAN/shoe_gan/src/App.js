import logo from './logo.svg';
import './App.css';
import { jsx, css } from "@emotion/react";
import React, { useState } from 'react';
/** @jsx jsx */
/** @jsxRuntime classic */

const shoes = [
  'https://media1.popsugar-assets.com/files/thumbor/a8EPLJKh8pae6CDVT6ZVAUeN3lY/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2018/01/22/867/n/1922729/01f9ba2eb749f455_netimgoUTZ8l/i/Adidas-Sneakers-Sale-2018.jpg',
  'https://www.surfstitch.com/on/demandware.static/-/Sites-ss-master-catalog/default/dw4342294a/images/BY3838BLK/BLACK-WHITE-KIDS-BOYS-ADIDAS-ORIGINALS-SNEAKERS-BY3838BLK_3.JPG'
]



const App = () => {

  const [idx, setShoe] = useState(0)
  console.log(idx)
  console.log(shoes[idx])

  return (
    <div className="App">
      <header className="App-header">
        <div className='Main'>
          <div>
            <img src={shoes[idx]}/>
          </div>
          <div>
            <button onClick={() => setShoe((idx + 1) % shoes.length)}>Like</button>
            <button onClick={() => setShoe((idx + 1) % shoes.length)}>Dislike</button>
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
