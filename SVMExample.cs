using UnityEngine;

public class SVMExample : MonoBehaviour
{
    void Start()
    {
        SVMModel svm = new SVMModel();
        // Add support vectors to the model. In a real scenario, these would be determined by training the SVM.
        svm.AddSupportVector(new Vector3(1, 2, 3), 0.5f);
        svm.AddSupportVector(new Vector3(4, 5, 6), 0.8f);

        Vector3 testPoint = new Vector3(2, 3, 4);
        float decision = svm.Decide(testPoint);
        Debug.Log(string.Format("Decision score for testPoint is: {0}", decision));
    }
}
