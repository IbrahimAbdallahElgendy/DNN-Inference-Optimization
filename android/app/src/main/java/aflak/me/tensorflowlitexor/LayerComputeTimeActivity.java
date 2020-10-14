package aflak.me.tensorflowlitexor;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.os.Trace;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import aflak.me.tensorflowlitexor.bean.LayerBean;
import aflak.me.tensorflowlitexor.util.Constant;

public class LayerComputeTimeActivity extends AppCompatActivity {
    private static final String TAG = "LayerComputeTime";
    private TextView tvTime;
    private TensorFlowInferenceInterface inferenceInterface;

    ArrayList<LayerBean> layerInfos;
    /**
     * 改变要执行的模型
     */
    private int RUN_MODEL_FLAG = Constant.VGG;
    private int input_layer_nums = 1;
    private int runNum = 100;
    /**
     * 不同模型的相关配置
     */
    private String pbModelPath;
    private String nodeJsonPath;
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_layer_compute_time);
        checkNeedPermissions();

        tvTime = findViewById(R.id.tv_ct);

        initModelProperties();
        /**
         * 加载冻结的 tensorflow  和节点信息json文件
         */
        Log.d(getClass().getSimpleName(), "Loading Model");
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), pbModelPath);
        Log.d(getClass().getSimpleName(), "Loaded Model");

        Gson gson = new Gson();
        String nodesJsonStr = getJson(nodeJsonPath, LayerComputeTimeActivity.this);
        layerInfos = gson.fromJson(nodesJsonStr, new TypeToken<ArrayList<LayerBean>>() {}.getType());
        Log.i(TAG,"layerInfos length " + layerInfos.size());

    }

    private void checkNeedPermissions(){
        //6.0以上需要动态申请权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            //多个权限一起申请
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.READ_EXTERNAL_STORAGE
            }, 1);
        }
    }

    private void initModelProperties() {
        switch(this.RUN_MODEL_FLAG){

            case Constant.GoogLeNet :
                pbModelPath = "GoogLeNet_model.pb";
                nodeJsonPath = "GoogLeNet_node.json";
                inputSize = 32 * 32 * 3;
                outputSize = 10;
                width = 32;
                height = 32;
                channel = 3;
                inputName = "input_1";
                outputName = "dense_5/Softmax";
                break;

            case Constant.VGG:
                pbModelPath = "VGG_model.pb" ;
                nodeJsonPath = "VGG_node.json";
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
                nodeJsonPath = "pix2pix_node.json";
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
                nodeJsonPath = "AlexNet_node.json";
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
                nodeJsonPath = "ResNet_node.json";
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
                nodeJsonPath = "InceptionV3_node.json";
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
                nodeJsonPath = "InceptionV4_node.json";
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
                nodeJsonPath = "InceptionResnetV2_node.json";
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
                nodeJsonPath = "SEResNeXt_node.json";
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
                nodeJsonPath = "MobileNet_node.json";
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
                nodeJsonPath = "Chairs_node.json";
                outputSize = 128*128*1;
                outputName = "conv2d_7/BiasAdd";
                input_layer_nums = 3;
                break;

            case Constant.DeepSpeech:
                pbModelPath = "DeepSpeech_model.pb";
                nodeJsonPath = "DeepSpeech_node.json";
                inputSize = 256 * 26;
                outputSize = 256 * 29;
                width = 256;
                height = 26;
                inputName = "input_1";
                outputName = "predictions/Reshape_1";
                break;

            case Constant.Seq2Seq:
                pbModelPath = "Seq2Seq_model.pb";
                nodeJsonPath = "Seq2Seq_node.json";
                outputSize = 10000;
                outputName = "predictions/truediv";
                break;

            case Constant.DQN:
                pbModelPath = "DQN_model.pb";
                nodeJsonPath = "DQN_node.json";
                inputSize = 80 * 80 * 4;
                outputSize = 18;
                width = 80;
                height = 80;
                channel = 4;
                inputName = "input_1";
                outputName = "predictions/BiasAdd";
                break;

            default :
                Log.e(getClass().getSimpleName(), "RUN_MODEL_FLAG is not right...");
        }

    }

    public void doGoogLeComputeTime(View view) {
        Thread thread = new Thread(new Runnable() {

            @Override
            public void run() {
                try {
                    String data = predict();
                    /**
                     * 保存文件
                     */
                    saveCtList(data);
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        });
        thread.start();
    }

    private String predict(){
        Log.i(TAG , "predict is called ...." );
        String data = "";
        double layersSum = 0;
        /**
         * 分步执行
         */
        int index = 0;
        for(LayerBean bean : layerInfos){
            // 第一层
            if(index < input_layer_nums){
                data = data + bean.layerName + "\t" + 0 + "\n";
                Log.i(TAG , bean.layerName + " " + 0.0 );
                index ++;
                continue;
            }
            /**
             * 预先执行
             */
            // step1: feed 数据
            Trace.beginSection("feed");
            for(int preIndex = 0; preIndex < bean.previousOpName.size() ;preIndex++){
                String sh = "";
                for(long n :bean.previousShape.get(preIndex) ){
                    sh += n + " ";
                }
                Log.i(TAG , index + ", " + bean.layerName + " feed " + preIndex +
                        " " + bean.previousOpName.get(preIndex) + " " + sh);
                float[] feddInput = new float[ bean.previousSize.get(preIndex) ];
                inferenceInterface.feed(bean.previousOpName.get(preIndex), feddInput,bean.previousShape.get(preIndex) );
            }
            Trace.endSection();

            // step2: 执行推理
            Trace.beginSection("run");
            inferenceInterface.run(new String[]{ bean.opName} );
            Trace.endSection();

            /**
             * 执行runNum次求平均
             */
            long sumTime = 0;
            for (int j = 0;j < runNum;j++){
                long begin ,end;

                // 执行当前层
                begin = System.nanoTime();//获取纳秒

                // 输入数据
                for(int preIndex = 0; preIndex < bean.previousOpName.size() ;preIndex++){
                    float[] feddInput = new float[ bean.previousSize.get(preIndex) ];
                    // feed(String inputName, float[] src, double... dims)
                    inferenceInterface.feed(bean.previousOpName.get(preIndex), feddInput,bean.previousShape.get(preIndex) );
                }


                inferenceInterface.run(new String[]{ bean.opName} );
                end = System.nanoTime();//获取纳秒

                // 统计时间
                sumTime += (end-begin);
            }
            // 计算平均值
            double meanTime = 1.0 * sumTime /runNum /1000000.0;
            // 保留4位小数
            meanTime = new BigDecimal(meanTime).setScale(4, BigDecimal.ROUND_HALF_UP).doubleValue();

            data = data + bean.layerName + "\t" + meanTime + "\n";
            Log.i(TAG , index + ", " + bean.layerName + " " + meanTime );
            layersSum += meanTime;
            index ++;

        }
        Log.i(TAG,"layersSum " + layersSum );
        /**
         * 更新UI
         */
        final double tvTimeNum = layersSum;
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tvTime.setText(String.valueOf(tvTimeNum));
            }
        });
        return data;
    }

    /**
     * 写文件到SDCard
     * @param data
     * @return
     */
    public static boolean saveCtList(String data) {

        try {
            // 判断当前的手机是否有sd卡
            String state = Environment.getExternalStorageState();

            if(!Environment.MEDIA_MOUNTED.equals(state)) {
                // 已经挂载了sd卡
                return false;
            }

            File sdCardFile = Environment.getExternalStorageDirectory();
            File file = new File(sdCardFile, "MobileNodeComputeTime.txt");

            FileOutputStream fos = new FileOutputStream(file);

//            String data = "";
//            for(int i = 0;i < ctList.size();i++){
//                data = data + ctList.get(i) + "\n";
//            }

            fos.write(data.getBytes());

            fos.flush();
            fos.close();
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    /**
     * 读取本地Json文件
     * @param fileName
     * @param context
     * @return
     */
    public static String getJson(String fileName, Context context){
        StringBuilder stringBuilder = new StringBuilder();
        try {
            InputStream is = context.getAssets().open(fileName);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(is));
            String line;
            while ((line=bufferedReader.readLine()) != null){
                stringBuilder.append(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return stringBuilder.toString();
    }

}
