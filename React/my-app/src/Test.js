import React, { useState } from 'react';

function Simple() {
    // Declare a new state variable, which we'll call "count"
    const [count, setCount] = useState(100);

    const decrement = () => {
        setCount(count - 1)
    }
  
    return (
      <div>
        <p>Count down from {count} !</p>
        <button onClick={decrement}>
          Click to count down
        </button>
      </div>
    );
  }

export default Simple;