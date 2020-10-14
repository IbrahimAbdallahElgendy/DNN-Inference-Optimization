package aflak.me.tensorflowlitexor;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import com.google.gson.Gson;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;

import aflak.me.tensorflowlitexor.bean.OutSizeBean;
import aflak.me.tensorflowlitexor.util.Constant;

public class LayerUpTimeActivity extends AppCompatActivity {
    private static final String TAG = "LayerUpTime";

    private TextView tvTime;
    /**
     * 改变要执行的模型
     */
    private int RUN_MODEL_FLAG = Constant.VGG;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        checkNeedPermissions();
        setContentView(R.layout.activity_layer_up_time);

        tvTime = findViewById(R.id.tv_ut);

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

    public void doGoogLeUpTime(View view) {
        Thread thread = new Thread(new Runnable() {
            int runNum = 200;
            @Override
            public void run() {
                /**
                 * 加载节点信息json文件
                 */
                Gson gson = new Gson();
                String jsonStr = "";
                OutSizeBean outSizeBean = null;
                switch(RUN_MODEL_FLAG){

                    case Constant.AlexNet :
                        jsonStr = getJson("AlexNet_outsize.json", LayerUpTimeActivity.this);
                        outSizeBean = gson.fromJson(jsonStr, OutSizeBean.class);
                        Log.i(TAG,"outSizeBean  length " + outSizeBean.layerOutSizes.size());
                        break;

                    case Constant.VGG:
                        jsonStr = getJson("VGG_outsize.json", LayerUpTimeActivity.this);
                         outSizeBean = gson.fromJson(jsonStr, OutSizeBean.class);
                        Log.i(TAG,"outSizeBean  length " + outSizeBean.layerOutSizes.size());
                        break;

                    default :
                        Log.e(getClass().getSimpleName(), "RUN_MODEL_FLAG is not right...");
                        break;
                }

                // 要上载的数据大小
                ArrayList<Integer> sizeList = outSizeBean.layerOutSizes;
                // 排序
                Collections.sort(sizeList);
                // 对应的上载时间
                ArrayList<Double> upTimeList = new ArrayList<Double>();
                double allUpTime = 0;
                // 开始测量
                try{
                    try{

                        for(Integer size : sizeList){
                            double sumTime = 0;
                            for(int i = 0; i < runNum ;i++){
                                // 准备要发送的数组
                                float [] arr = new float[ size];

                                // t0 记录发送数据开始时间
                                double t0 = System.nanoTime();//获取纳秒
                                /**
                                 * 连接上服务器
                                 * TODO: 在此处相应地更改服务器IP和PORT.
                                 */
                                Socket socket = new Socket("192.168.0.177", 8000);
//                        PrintWriter outToServer = new PrintWriter(
//                                new OutputStreamWriter(socket.getOutputStream())
//                        );
                                DataOutputStream outToServer = new DataOutputStream(socket.getOutputStream());
                                /**
                                 * 将float数组编码为字符串
                                 */
                                byte[] bytearr = FloatArray2ByteArray(arr);
//                                Log.i(TAG,"bytearr len "+bytearr.length);
//                                String encodedString = Base64.encodeToString(bytearr, Base64.DEFAULT);
                                /**
                                 * 发送
                                 */
//                                outToServer.print(encodedString);
                                outToServer.write(bytearr);

                                // 关闭资源
                                outToServer.flush();
                                outToServer.close();
                                socket.close();

                                // t1 记录发送数据结束时间
                                double t1 = System.nanoTime();//获取纳秒

                                // 计时
                                double upTime = (t1 - t0)/(1000000.0);
                                sumTime += upTime;
                                Log.i(TAG, size + " " + upTime);

                                Thread.sleep(50);    //延时50ms
                            }
                            double meanTime = sumTime/runNum;
                            meanTime = new BigDecimal(meanTime).setScale(4, BigDecimal.ROUND_HALF_UP).doubleValue();
                            upTimeList.add(meanTime);
                            allUpTime += meanTime;
                        }

                    }catch(IOException e){
                        e.printStackTrace();
                        Log.e(TAG,e.toString());
                    }
                }catch (Exception e){
                    e.printStackTrace();
                    Log.e(TAG,e.toString());
                }


                /**
                 * 开始保存数据
                 */
                saveUtList(LayerUpTimeActivity.this, sizeList,upTimeList);

                /**
                 * 更新UI
                 */
                final double tvTimeNum = allUpTime;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tvTime.setText(String.valueOf(tvTimeNum));
                    }
                });
            }
        });

        thread.start();
    }

    /**
     * 写文件到SDCard
     * @param context
     * @param sizeList
     * @param upTimeList
     * @return
     */
    public static boolean saveUtList(Context context, ArrayList<Integer> sizeList,ArrayList<Double> upTimeList) {
        Log.i(TAG,"saveUtList called ....");

        try {
            // 判断当前的手机是否有sd卡
            String state = Environment.getExternalStorageState();

            if(!Environment.MEDIA_MOUNTED.equals(state)) {
                // 已经挂载了sd卡
                return false;
            }

            File sdCardFile = Environment.getExternalStorageDirectory();
            File file = new File(sdCardFile, "MobileNodeUploadTime.txt");

            FileOutputStream fos = new FileOutputStream(file);

            String data = "";
            for(int i = 0;i < sizeList.size();i++){
                data = data + sizeList.get(i) + "\t" + upTimeList.get(i) + "\n";
            }

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

    public static byte[] FloatArray2ByteArray(float[] values){
        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);

        for (float value : values){
            buffer.putFloat(value);
        }

        return buffer.array();
    }
}
