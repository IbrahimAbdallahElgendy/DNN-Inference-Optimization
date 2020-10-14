package aflak.me.tensorflowlitexor;

import android.os.AsyncTask;
import android.util.Base64;
import android.util.Log;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.ArrayList;

public class SendMessage extends AsyncTask<Void, Void, Void> {
    private Exception exception;
    private float[] arr;
    ArrayList<Long> upTimeList = null;

    public SendMessage(float[] arr){
        this.arr = arr;
    }

    public SendMessage(float[] arr,ArrayList<Long> upTimeList){
        this.arr = arr;
        this.upTimeList = upTimeList;
    }

    @Override
    protected Void doInBackground(Void... params){
        // t0 记录发送数据开始时间
        long t0 = System.currentTimeMillis();

        /**
         * 将float数组编码为字符串
         */
        byte[] bytearr = FloatArray2ByteArray(arr);
        String encodedString = Base64.encodeToString(bytearr, Base64.DEFAULT);

        try{
            try{
                /**
                 * TODO: 在此处相应地更改服务器IP和PORT.
                 */

                Socket socket = new Socket("192.168.0.177", 8000);
                PrintWriter outToServer = new PrintWriter(
                        new OutputStreamWriter(socket.getOutputStream())
                );
                outToServer.print(encodedString);
                outToServer.flush();
                socket.close();

            }catch(IOException e){
                e.printStackTrace();
            }
        }catch (Exception e){
            this.exception = e;
            return null;
        }
        // t1 记录发送数据结束时间
        long t1 = System.currentTimeMillis();
        /**
         * "INFO0222" 指示发送消息所花费的时间.
         */
        Log.i("INFO0222" , Long.toString(t1-t0));

        if(upTimeList != null){
            upTimeList.add(t1 - t0);
        }
        return null;

    }

    public static byte[] FloatArray2ByteArray(float[] values){
        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);

        for (float value : values){
            buffer.putFloat(value);
        }

        return buffer.array();
    }
}
