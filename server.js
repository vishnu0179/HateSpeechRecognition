const express = require('express')
const app = express()
const bodyParser = require('body-parser')
let spawn = require('child_process').spawn;


app.use('/',express.static(__dirname))
app.use(bodyParser.urlencoded({extended: false}))

app.post('/hashtag', (req,res)=>{
    let HTprocess = spawn('python', ["./HateSpeechRecognition.py", req.body.hashtag , req.body.tweetCount ])

    HTprocess.stdout.on('data', (data)=> {
        res.send(data.toString())
    })
})

app.post('/useranalysis', (req,res)=>{
    let UAprocess = spawn('python3', ["./fetchtweets.py", req.body.name])
    console.log(req.body.name)
    console.log('spawned the process')
    UAprocess.stdout.on('data', (data)=> {
        console.log('fetched data')
        res.send(data.toString())
    })
})

app.get('/tweets',(req,res)=>{
    res.redirect('/tweets.txt')
})

app.listen(8080,()=>{
    console.log('Server started on port 8080')
})