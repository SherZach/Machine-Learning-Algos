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
      <header className="App-header">
      <img src="Ai_core_logo.png" className="App-logo" alt="AICoreLogo" css={css`height:50; width: 50`}/>
      </header>
      <body css={css`background-color: #282c34; min-height: 100vh`}>
        <div css={css`color: dodgerblue; font-size: 40px`} className="Text-spin">
          {name}
        </div>
      </body>
    </div>
  );
}

export default App;
