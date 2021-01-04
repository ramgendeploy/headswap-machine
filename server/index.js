const express = require('express')
const nodemailer = require('nodemailer')

const app = express()
const port = 3000

app.use(express.json());

app.get('/', (req, res)=>{
  res.send('Hello World!')
  // res.status(403);
})


app.post('/face', (req, res)=>{

  console.log(req.body)

  res.send(req.body)

  // res.send('Data recived')
})



app.listen(port, ()=> {
  console.log(`Example app listening at http://localhost:${port}`)
})
