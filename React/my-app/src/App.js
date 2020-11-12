import logo from './logo.svg';
import './App.css';
import { jsx, css } from "@emotion/core"
/** @jsx jsx */
/** @jsxRuntime classic */

const name = 'Weeeeeeee!!!!!'
// https://www.theaicore.com/wp-content/uploads/2020/11/Screenshot-2020-11-01-at-15.13.05.png

const App = () => {
  return (
    <div className="App">
      <header>
        <div className="container_pink">
          <div className="item">
          </div>
          <div className="item">
          </div>
        </div>   
      </header>
      <body>
        <div classname="big_box">
          <div className="bottom_box">
            <div className="pink_bot_box">
              <div className="green_box">box</div>
              <div className="green_box">box</div>
            </div>
            <div className="pink_bot_box">
              <div className="blue_box">box</div>
              <div className="blue_box">box</div>
            </div>
          </div>
          <div className="bottom_blue_box">
            box
          </div>
        </div>
      </body>
    </div>
  );
}

export default App;
