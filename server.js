const express = require('express')
const app = express()
const bodyParser = require('body-parser')
let spawn = require('child_process').spawn;


app.use('/',express.static(__dirname))
app.use(bodyParser.urlencoded({extended: false}))

app.post('/result', (req,res)=>{
    let process = spawn('python', ["./abc.py", req.body.first_name, req.body.last_name ])

    process.stdout.on('data', (data)=> {
        res.send(data.toString())
    })
})

app.get('/tweets',(req,res)=>{
    res.redirect('/tweets.txt')
})

app.listen(3000,()=>{
    console.log('Server started on port 3000')
})