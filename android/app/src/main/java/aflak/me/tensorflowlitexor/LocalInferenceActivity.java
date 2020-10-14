package aflak.me.tensorflowlitexor;

import android.os.Trace;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.contrib.android.RunStats;

import java.math.BigDecimal;

import aflak.me.tensorflowlitexor.util.Constant;

public class LocalInferenceActivity extends AppCompatActivity {

    private static final String TAG = "LocalInferenceActivity";

    private TextView tvTime;
    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * 不同模型的相关配置
     */
    private String pbModelPath;
    private int inputSize;
    private int outputSize;
    //数据的维度
    private long width;
    private long height;
    private long channel ;
    //模型中输出变量的名称
    private String inputName ;
    //模型中输出变量的名称
    private String outputName ;

    /**
     * 改变要执行的模型
     */
    private int RUN_MODEL_FLAG = Constant.VGG;
    private int runNum = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_local_inference);

        tvTime = findViewById(R.id.tv_mean_local_infer_time);

        initModelProperties();

        /**
         * 加载冻结的 tensorflow 模型
         */
        Log.d(getClass().getSimpleName(), "Loading Model");
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), pbModelPath);
        Log.d(getClass().getSimpleName(), "Loaded Model");
    }

    private void initModelProperties() {
        switch(this.RUN_MODEL_FLAG){

            case Constant.GoogLeNet :
                pbModelPath = "GoogLeNet_model.pb";
                inputSize = 32 * 32 * 3;
                outputSize = 1000;
                width = 32;
                height = 32;
                channel = 3;
                inputName = "input_1";
                outputName = "dense_5/Softmax";
                break;

            case Constant.VGG:
                pbModelPath = "VGG_model.pb" ;
                inputSize = 64 * 64 * 3;
                outputSize = 1000;
                width = 64;
                height = 64;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.Pix2Pix:
                pbModelPath = "pix2pix_model.pb";
                inputSize = 256 * 256 * 3;
                outputSize = 256 * 256 * 3;
                width = 256;
                height = 256;
                channel = 3;
                inputName = "input_1";
                outputName = "conv2d_transpose_1/Tanh";
                break;

            case Constant.AlexNet:
                pbModelPath = "AlexNet_model.pb";
                inputSize = 224 * 224 * 3;
                outputSize = 1000;
                width = 224;
                height = 224;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.ResNet:
                pbModelPath = "ResNet_model.pb";
                inputSize = 224 * 224 * 3;
                outputSize = 1000;
                width = 224;
                height = 224;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.InceptionV3:
                pbModelPath = "InceptionV3_model.pb";
                inputSize = 299 * 299 * 3;
                outputSize = 1000;
                width = 299;
                height = 299;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.InceptionV4:
                pbModelPath = "InceptionV4_model.pb";
                inputSize = 299 * 299 * 3;
                outputSize = 1000;
                width = 299;
                height = 299;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.InceptionResNet:
                pbModelPath = "InceptionResnetV2_model.pb";
                inputSize = 299 * 299 * 3;
                outputSize = 1000;
                width = 299;
                height = 299;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.SEResNeXt:
                pbModelPath = "SEResNeXt_model.pb";
                inputSize = 32 * 32 * 3;
                outputSize = 10;
                width = 32;
                height = 32;
                channel = 3;
                inputName = "input_1";
                outputName = "predictions/Softmax";
                break;

            case Constant.MobileNet:
                pbModelPath = "MobileNet_model.pb";
                inputSize = 224 * 224 * 3;
                outputSize = 1000;
                width = 224;
                height = 224;
                channel = 3;
                inputName = "input_1";
                outputName = "reshape_2/Reshape";
                break;

            case Constant.Chairs:
                pbModelPath = "Chairs_model.pb";
                outputSize = 128*128*1;
                outputName = "conv2d_7/BiasAdd";
                break;

            case Constant.DeepSpeech:
                pbModelPath = "DeepSpeech_model.pb";
                inputSize = 256 * 26;
                outputSize = 256 * 29;
                width = 256;
                height = 26;
                inputName = "input_1";
                outputName = "predictions/Reshape_1";
                break;

            case Constant.Seq2Seq:
                pbModelPath = "Seq2Seq_model.pb";
                outputSize = 20*10000;
                outputName = "predictions/truediv";
                break;

            case Constant.DQN:
                pbModelPath = "DQN_model.pb";
                inputSize = 80 * 80 * 4;
                outputSize = 18;
                width = 80;
                height = 80;
                channel = 4;
                inputName = "input_1";
                outputName = "predictions/BiasAdd";
                break;

            case Constant.VAE:
                pbModelPath = "VAE_model.pb";
                outputSize = 1024;//encoder_input
                outputName = "predictions/Sigmoid";
                break;

            default :
                Log.e(getClass().getSimpleName(), "RUN_MODEL_FLAG is not right...");
        }

    }

    public void doBeginLocalInference(View view){

        Thread thread = new Thread(new Runnable() {

            @Override
            public void run() {
//                // 预先执行两次
                localInferOnce();
                // 正式开始统计
                double localInferTimeSum = 0;
                for(int i = 0; i < runNum ;i++){
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    double localInferTime = localInferOnce();
                    localInferTimeSum += localInferTime;
                }
                // 求平均，保留4位小数
                double tmpMean = localInferTimeSum/runNum;
                tmpMean = new BigDecimal(tmpMean).setScale(4, BigDecimal.ROUND_HALF_UP).doubleValue();

                Log.i("INFO LocalInfer mean " , tmpMean + "" );
                /**
                 * 更新UI
                 */
                final double meanLocalInferTime = tmpMean;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tvTime.setText(String.valueOf(meanLocalInferTime));
                    }
                });
            }
        });
        thread.start();
    }

    private double localInferOnce(){
        // t0 表示整体开始的时间
        long t0 = System.nanoTime();//获取纳秒
        long t1 = 0;

        // step2: 将照片转换成float数组
        float[] inputData = new float[this.inputSize] ;

        t1 = System.nanoTime();//获取纳秒

        double data_fetch_time = (t1 - t0)/(1000000.0);

        // step3: 检查网络是否连接
        // 步骤1：更新t0 => 本地推理开始时间
        t0 = System.nanoTime();//获取纳秒

        // 步骤2：本地推理整个模型
//        float[] output = predict(inputData);
        float[] output;
        try {
            switch(this.RUN_MODEL_FLAG){
                case Constant.DeepSpeech:
                    output = predictDeepSpeech(inputData);
                    break;

                case Constant.Seq2Seq:
                    output = predictSeq2Seq();
                    break;

                case Constant.Chairs:
                    output = predictChairs();

                    break;

                case Constant.VAE:
                    output = predictVAE();
                    break;

                default:
                    output = predict(inputData);
                    break;
            }

        }catch (Exception e){
            e.printStackTrace();
        }
        // 步骤3：更新t1 => 本地推理结束时间
        t1 = System.nanoTime();//获取纳秒

        double localInferTime = (t1 - t0)/(1000000.0);

        // 日志：INFO LocalInfer =  数据获取时间,本地推理时间
        Log.i("INFO LocalInfer" , localInferTime + ", " + data_fetch_time);

        return localInferTime;
    }

    /** 本地推理整个模型. */
    private float[] predict(float[] input){
        float output[] = new float[outputSize];

        // step1: feed 数据
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, input, 1,  width, height, channel);
        Trace.endSection();

        // step2: 执行推理
        Trace.beginSection("run");
        inferenceInterface.run(new String[]{outputName});
        Trace.endSection();

        // step3: 获取推理结果
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, output);
        Trace.endSection();

        return output;// return prediction
    }

    private float[] predictChairs(){
        float output[] = new float[outputSize];

        int class_len = 809;
        int view_len = 4;
        int transf_len = 12;
        float[] classInputData = new float[class_len] ;
        float[] viewInputData = new float[view_len] ;
        float[] transfInputData = new float[transf_len] ;

        // step1: feed 数据
        Trace.beginSection("feed");
        inferenceInterface.feed("class", classInputData, 1,  class_len);
        inferenceInterface.feed("transf_param", transfInputData, 1,  transf_len);
        inferenceInterface.feed("view", viewInputData ,1,  view_len);
        Trace.endSection();

        // step2: 执行推理
        Trace.beginSection("run");
        inferenceInterface.run(new String[]{outputName});
        Trace.endSection();

        // step3: 获取推理结果
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, output);
        Trace.endSection();

        return output;// return prediction
    }

    private float[] predictSeq2Seq(){
        float output[] = new float[outputSize];

        float[] inputData1 = new float[20] ;
        float[] inputData2 = new float[20] ;
        // step1: feed 数据
        Trace.beginSection("feed");
        inferenceInterface.feed("input_1", inputData1, 1,  20);
        inferenceInterface.feed("input_2", inputData2, 1,  20);
        Trace.endSection();

        // step2: 执行推理
        Trace.beginSection("run");
        inferenceInterface.run(new String[]{outputName});
        Trace.endSection();

        // step3: 获取推理结果
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, output);
        Trace.endSection();

        return output;// return prediction
    }

    private float[] predictVAE(){
        float output[] = new float[outputSize];

        float[] inputData = new float[1024] ;
        // step1: feed 数据
        Trace.beginSection("feed");
        inferenceInterface.feed("encoder_input", inputData, 1,  1024);
        Trace.endSection();

        // step2: 执行推理
        Trace.beginSection("run");
        inferenceInterface.run(new String[]{outputName});
        Trace.endSection();

        // step3: 获取推理结果
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, output);
        Trace.endSection();

        return output;// return prediction
    }

    private float[] predictDeepSpeech(float[] input){
        float output[] = new float[outputSize];

        // step1: feed 数据
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, input, 1,  width, height);
        Trace.endSection();

        // step2: 执行推理
        Trace.beginSection("run");
        inferenceInterface.run(new String[]{outputName});
        Trace.endSection();

        // step3: 获取推理结果
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, output);
        Trace.endSection();

        return output;// return prediction
    }
//    private float[] predict(float[] input){
//
//        float output[] = new float[outputSize];
//        // step1: feed 数据
//        inferenceInterface.feed(inputName, input, 1,  width, height, channel);
//        // step2: 执行推理
//        inferenceInterface.run(new String[]{outputName});
//        // step3: 获取推理结果
//        inferenceInterface.fetch(outputName, output);
//        return output;  // return prediction
//    }

}
