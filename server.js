const express = require('express')
const app = express()
const bodyParser = require('body-parser')
let spawn = require('child_process').spawn;

const exphbs = require('express-handlebars')

app.use('/',express.static(__dirname))
app.use(bodyParser.urlencoded({extended: false}))

app.set('views', __dirname + '/views')
app.engine('handlebars', exphbs())
app.set('view engine','handlebars')

app.get('/useranalysis', (req, res)=>{
    res.render('useranalysis',{
        layout:false,
        display: 'none'
    })
})

app.get('/hashtag', (req, res)=>{
    res.render('hashtag',{
        layout:false,
        display: 'none'
    })
})


app.post('/hashtag', (req,res)=>{
    let HTprocess = spawn('python3', ["./Hashtag.py", req.body.hashtag , req.body.tweetCount ])
    console.log(req.body.hashtag)
    console.log(req.body.tweetCount);
    
    HTprocess.stdout.on('data', (data)=> {
        console.log('fetched data')
        res.render('hashtag',{
            layout:false,
            display: 'inline-block',
            status: 'Tweets Fetched',
            imgPath:'../plot.jpeg'
        })
    })
})

app.post('/userpredict', (req,res)=>{
    let UAprocess = spawn('python3', ["./fetchtweets.py", req.body.name])
    console.log(req.body.name)
    console.log('spawned the process')
    UAprocess.stdout.on('data', (data)=> {
        console.log('fetched data')
        res.render('useranalysis', {
            layout:false,
            status : 'Tweets Fetched',
            imgPath : '../plot.jpeg',
            display: 'inline-block'
        })
    })
})

/*app.get('/predict',(req,res)=>{
    let predictProcess = spawn('python3',["./predict.py"])
    res.write('Predicting')
    predictProcess.stdout.on('data',(data)=>{
        res.send()
    })
})*/

/*app.get('/',(req,res)=>{
    res.render('index')
})*/

app.listen(8080,()=>{
    console.log('Server started on port 8080')
})