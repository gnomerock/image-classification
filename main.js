const tf = require('@tensorflow/tfjs-node')

const express = require('express')
const bodyParser = require('body-parser')
const fileUpload = require('express-fileupload')

const mobilenet = require('@tensorflow-models/mobilenet');

const app = express()
const PORT = 3000

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(fileUpload({
  limits: { fileSize: 50 * 1024 * 1024 },
}));

app.get('/', (req, res) => {
  res.send('[+] Hello, This is image classification API')
})

app.post('/classify', async (req, res) => {
  
  let img = tf.node.decodeImage(req.files.image.data)
  
  try {
    const model = await mobilenet.load({
      version: 1,
      alpha: 1
    });
    const predictions = await model.classify(img);
    res.json({status: true, predictions})
  } catch (error) {
    console.error(error)
    res.json({status: false, error})
  }
  
})

app.listen(PORT , () => {
  console.log(`[+] Image classification API start on port: ${PORT}`)
})